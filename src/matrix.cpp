/**
   \example matrix.cpp
   matrix multiplication example program, demonstrating the
   use of tensor core hardware acceleration
 */

#include "common/matrix.hpp"
#include "common/output.hpp"
#include <cassert>
#include <chrono>
#include <goopax>
#include <goopax_draw/types.h>
#include <goopax_extra/output.hpp>
#include <goopax_extra/param.hpp>
#include <goopax_extra/random.hpp>
#include <goopax_extra/types.hpp>
#include <random>

using namespace Eigen;
using namespace std::chrono;
using namespace goopax;
using namespace std;

PARAMOPT<bool> VERB("verb", 0);

// Matrix sizes. Can be specified as command line arguments. See matrix --help
PARAMOPT<unsigned int> M("m", 4096);
PARAMOPT<unsigned int> N("n", 4096);
PARAMOPT<unsigned int> K("k", 4096);

// Use this workgroup size. If 0, a suitable value will be chosen.
PARAMOPT<unsigned int> LOCAL_SIZE("local_size", 0);

PARAMOPT<unsigned int> TILE_M("tile_m", 0);
PARAMOPT<unsigned int> TILE_N("tile_n", 0);
PARAMOPT<unsigned int> TILE_K("tile_k", 0);

PARAMOPT<bool> USE_BULK_COPY("use_bulk_copy", true);
PARAMOPT<bool> USE_ASYNC_COPY("use_async_copy", true);

template<typename func_t>
auto static_kernel(goopax_device device, func_t code)
{
    using kernel_t = decltype(kernel(device, code));

    static std::mutex Mutex;
    static std::unordered_map<detail::goopax_device_impl*, kernel_t> kernel_cache;

    kernel_t k;
    {
        std::lock_guard lock(Mutex);
        auto p = kernel_cache.find(device.get_impl());
        if (p == kernel_cache.end())
        {
            p = kernel_cache.emplace(device.get_impl(), kernel_t(device, code)).first;

            detail::release(device.get_impl());
            device.at_device_cleanup(
                [](void* data) {
                    detail::goopax_device_impl* device = static_cast<detail::goopax_device_impl*>(data);
                    detail::retain(device);
                    std::lock_guard lock(Mutex);
                    kernel_cache.erase(device);
                },
                device.get_impl());
        }
        k = p->second;
    }
    return k;
}

template<typename rnd_t>
rnd_t& static_rnd(goopax_device device)
{
    static std::unordered_map<detail::goopax_device_impl*, rnd_t> rnd_cache;

    static std::mutex Mutex;
    std::lock_guard lock(Mutex);

    auto r = rnd_cache.find(device.get_impl());
    if (r == rnd_cache.end())
    {
        std::random_device rd;
        r = rnd_cache.emplace(device.get_impl(), rnd_t(device, device.default_global_size_max(), rd())).first;
        detail::release(device.get_impl());
        device.at_device_cleanup(
            [](void* data) {
                detail::goopax_device_impl* device = static_cast<detail::goopax_device_impl*>(data);
                detail::retain(device);
                std::lock_guard lock(Mutex);
                rnd_cache.erase(device);
            },
            device.get_impl());
    }
    return r->second;
}

template<typename T>
void fill_random(buffer<T>& a)
{
    goopax_device device = a.get_device();

    static_kernel(device, [device](resource<T>& a) {
        WELL512_lib rndlib(static_rnd<WELL512_data>(device));
        for_each_global(a,
                        [&](auto& x) { x = static_cast<typename make_gpu<T>::type>(rndlib.gaussian_distribution()); });
    })(a);
}

template<typename T>
void make_sparse(buffer<T>& a, const matrix::matrix_support_info::sparse_t& sparse)
{
    goopax_device device = a.get_device();

    static_kernel(device, [&sparse](resource<T>& a) {
        xoshiro128_lib rndlib(static_rnd<xoshiro128_data>(get_current_build_device()));
        gpu_for_global(a.begin(), a.end(), sparse.blocksize, [&](auto p) {
            gpu_uint bits = rndlib.next() & ((1 << sparse.blocksize / sparse.vector_width) - 1);
            gpu_while(popcount(bits) != sparse.non_null_per_block / sparse.vector_width)
            {
                bits = rndlib.next() & ((1 << sparse.blocksize / sparse.vector_width) - 1);
            }
            for (unsigned int bit = 0; bit < sparse.blocksize / sparse.vector_width; ++bit)
            {
                gpu_if((bits & (1 << bit)) == 0)
                {
                    for (unsigned int sub = 0; sub < sparse.vector_width; ++sub)
                    {
                        p[bit * sparse.vector_width + sub] = 0;
                    }
                }
            }
        });
    })(a);
}

template<typename src_ptr_t, typename dest_ptr_t>
void parallel_copy(src_ptr_t src_orig, unsigned int N_orig, dest_ptr_t dest_orig, bool use_async)
{
    using value_t = typename goopax_remove_pointer<typename make_cpu<dest_ptr_t>::type>::type;
    using uint_type = typename change_gpu_mode<uint32_t, value_t>::type;

    assert(N_orig * bitsize<value_t>::value % bitsize<uint_type>::value == 0);
    unsigned int N = N_orig * bitsize<value_t>::value / bitsize<uint_type>::value;
    auto src = reinterpret<gpu_type<const uint_type*>>(src_orig);
    auto dest = reinterpret<typename make_gpu_pointer<uint_type, memory::threadgroup>::type>(dest_orig);

    unsigned int offset = 0;
    for (unsigned int burst = 4; burst != 0; burst /= 2)
    {
        while ((N - offset) % (burst * local_size()) == 0 && N - offset >= burst * local_size())
        {
            assert(N >= offset);
            for (unsigned int sub = 0; sub < burst; ++sub)
            {
                if (use_async)
                {
                    async_copy(src + local_id() * burst + (offset + sub), dest + local_id() * burst + (offset + sub));
                }
                else
                {
                    dest[local_id() * burst + (offset + sub)] = src[local_id() * burst + (offset + sub)];
                }
            }
            offset += burst * local_size();
        }
    }
    if (N - offset != 0)
    {
        gpu_if(local_id() < N - offset)
        {
            if (use_async)
            {
                async_copy(src + local_id() + offset, dest + local_id() + offset);
            }
            else
            {
                dest[local_id() + offset] = src[local_id() + offset];
            }
        }
    }
}

template<typename a_float_type,
         typename b_float_type,
         typename c_float_type,
         typename a_input_type,
         typename b_input_type,
         bool is_sparse,
         bool with_block_scaling,
         typename bs_type = typename change_gpu_mode<precision::fp8ue8m0, a_float_type>::type>
class Matmul
{
public:
    using a_float_type_nodebug = typename unrangetype<a_float_type>::type;
    using b_float_type_nodebug = typename unrangetype<b_float_type>::type;
    using c_float_type_nodebug = typename unrangetype<c_float_type>::type;

    using matrix_a_t = typename std::
        conditional<is_sparse, matrix::sparse_matrix<a_float_type>, matrix::warp_matrix<a_float_type>>::type;

    goopax_device device;

    const matrix::matrix_support_info* mi = nullptr;
    Tuint sparse_md_tile_size;
    Tuint tile_m;
    Tuint tile_n;
    Tuint tile_k;
    Tuint ls;

    buffer<a_float_type> Abuf_fast;
    buffer<b_float_type> Bbuf_fast;
    buffer<Tuint> sparse_md;
    buffer<bs_type> A_bs_buf_fast;
    buffer<bs_type> B_bs_buf_fast;

    kernel<void(const buffer<a_input_type>& A)> prepare_A;
    kernel<void(const buffer<a_input_type>& A)> prepare_A_from_dense;
    kernel<void(const buffer<b_input_type>& B)> prepare_B;
    kernel<void(const buffer<bs_type>& A_bs)> prepare_A_bs;
    kernel<void(const buffer<bs_type>& B_bs)> prepare_B_bs;
    kernel<void(buffer<c_float_type>& D)> multiply;

    template<bool from_dense, typename input_T, typename T>
    auto prepare_func(buffer<T>& Mbuf_fast,
                      buffer<Tuint>* M_md,
                      unsigned int rows,
                      unsigned int cols,
                      unsigned int tile_rows,
                      unsigned int tile_cols,
                      matrix::use_t use,
                      const matrix::matrix_support_info* mi)
    {
        assert(mi);
        return [rows, cols, tile_rows, tile_cols, use, mi, &Mbuf_fast, M_md, this](const resource<input_T>& Mbuf) {
            assert(local_size() == mi->Nthreads);
            gpu_for_group(0, (rows / tile_rows) * (cols / tile_cols), [&](gpu_uint tile) {
                gpu_uint tr;
                gpu_uint tc;

                // Arrange the tiles in row_major for matrix_b and col_major for matrix_a.
                if (use == matrix::matrix_b || use == matrix::matrix_block_scaling_b)
                {
                    tr = tile / (cols / tile_cols);
                    tc = tile % (cols / tile_cols);
                }
                else if (use == matrix::matrix_a || use == matrix::matrix_block_scaling_a)
                {
                    tc = tile / (rows / tile_rows);
                    tr = tile % (rows / tile_rows);
                }
                else
                    abort();

                if (is_sparse && use == matrix::matrix_a)
                {
                    if constexpr (is_sparse)
                    {
                        matrix::sparse_matrix<typename unrangetype<T>::type> M(tile_rows, tile_cols);
                        M.set_info(mi);
                        M.set_use(use);
                        if constexpr (from_dense)
                        {
                            M.load_cast_construct_from_dense(Mbuf.begin() + (tr * tile_rows * cols + tc * tile_cols),
                                                             cols);
                        }
                        else
                        {
                            M.load_cast(Mbuf.begin() + (tr * tile_rows * cols + tc * tile_cols) / mi->sparsity(use),
                                        matrix::row_major,
                                        cols);
                        }
                        M.store(Mbuf_fast.begin() + tile * tile_rows * tile_cols / mi->sparsity(use),
                                matrix::layout_optimal);
                        if (M_md)
                        {
                            M.store_metadata(M_md->begin() + tile * sparse_md_tile_size);
                        }
                    }
                }
                else
                {
                    matrix::warp_matrix<typename unrangetype<T>::type> M(tile_rows, tile_cols);
                    M.load_cast(Mbuf.begin() + tr * tile_rows * cols + tc * tile_cols, matrix::row_major, cols);
                    M.set_info(mi);
                    M.set_use(use);
                    M.store(Mbuf_fast.begin() + tile * tile_rows * tile_cols, matrix::layout_optimal);
                }
            });
        };
    }

    Matmul(goopax_device device0,
           unsigned int m,
           unsigned int n,
           unsigned int k,
           unsigned int force_tile_m,
           unsigned int force_tile_n,
           unsigned int force_tile_k,
           unsigned int force_ls)
        : device(device0)
    {
        tile_m = 64;
        tile_n = 128;
        tile_k = 16;
        ls = 256;

        if (bitsize<c_float_type>::value >= 32)
        {
            tile_m = 32;
            tile_n = 64;
        }
        if (bitsize<a_float_type>::value <= 8)
        {
            tile_k = 32;
        }
        if (std::is_same_v<a_float_type, precision::fp4e2m1>)
        {
            tile_k = 64;
        }
        if (std::is_same_v<a_float_type, double>)
        {
            tile_m = 8;
            tile_n = 16;
        }
        if (is_sparse)
        {
            if (bitsize<c_float_type>::value == 16 && bitsize<a_float_type>::value == 16)
            {
            }
            else if (bitsize<a_float_type>::value == 32)
            {
            }
            else
            {
                tile_k *= 2;
            }
        }

        if (force_tile_m)
            tile_m = force_tile_m;
        if (force_tile_n)
            tile_n = force_tile_n;
        if (force_tile_k)
            tile_k = force_tile_k;
        if (force_ls)
            ls = force_ls;

        if (m % tile_m != 0 || n % tile_n != 0 || k % tile_k != 0)
        {
            throw std::runtime_error("big matrix sizes must be multiples of tile sizes");
        }

        Tuint ws;
        for (const matrix::matrix_support_info* mi_try = device.get_matrix_support_table(); mi_try;
             mi_try = mi_try->next)
        {
            if (mi_try->type_enum_a == type_enum<a_float_type_nodebug>::value
                && mi_try->type_enum_b == type_enum<b_float_type_nodebug>::value
                && mi_try->type_enum_c == type_enum<c_float_type_nodebug>::value && mi_try->is_sparse() == is_sparse
                && mi_try->with_block_scaling() == with_block_scaling && tile_m % mi_try->mnk[0] == 0
                && tile_n % mi_try->mnk[1] == 0 && tile_k % mi_try->mnk[2] == 0 && ls % mi_try->Nthreads == 0)
            {
                mi = mi_try;
                ws = mi_try->Nthreads;
                break;
            }
        }
        if (mi == nullptr)
        {
            return;
        }
        // cout << "using mi=" << *mi << endl;
        Abuf_fast.assign(device, m * k / mi->sparsity(matrix::matrix_a));
        Bbuf_fast.assign(device, k * n);

        multiply.assign(
            device,
            [&](resource<c_float_type>& Dbuf) {
                if constexpr (is_sparse)
                {
                    matrix_a_t A(tile_m, tile_k);
                    A.set_info(mi);
                    sparse_md_tile_size = A.sparse_metadata().size() * mi->Nthreads;

                    if (sparse_md.empty())
                    {
                        sparse_md.assign(device, (m / tile_m) * (k / tile_k) * sparse_md_tile_size);
                    }
                }

                if (ls == ws)
                {
                    gpu_for_group(0, (m / tile_m) * (n / tile_n), [&](gpu_uint tile) {
                        gpu_uint tm = tile / (n / tile_n);
                        gpu_uint tn = tile % (n / tile_n);

                        matrix::warp_matrix<typename make_cpu<c_float_type_nodebug>::type> D(tile_m, tile_n);
                        D.fill(0);

                        gpu_for(0, (k / tile_k), [&](gpu_uint tk) {
                            matrix_a_t A(tile_m, tile_k);
                            A.set_info(mi);
                            A.load(Abuf_fast.begin()
                                       + (tk * (m / tile_m) + tm) * tile_m * tile_k / mi->sparsity(matrix::matrix_a),
                                   matrix::layout_optimal);
                            if constexpr (is_sparse)
                            {
                                A.load_metadata(sparse_md.begin() + (tk * (m / tile_m) + tm) * sparse_md_tile_size);
                            }
                            matrix::warp_matrix<typename make_cpu<b_float_type_nodebug>::type> B(
                                tile_k,
                                tile_n,
                                Bbuf_fast.begin() + (tk * (n / tile_n) + tn) * tile_k * tile_n,
                                matrix::layout_optimal);

                            if constexpr (with_block_scaling)
                            {
                                matrix::warp_matrix<bs_type> A_bs(tile_m,
                                                                  tile_k / mi->block_scaling.block_width,
                                                                  A_bs_buf_fast.begin()
                                                                      + (tm * (k / tile_k) + tk) * tile_m * tile_k
                                                                            / mi->block_scaling.block_width,
                                                                  matrix::layout_optimal);
                                matrix::warp_matrix<bs_type> B_bs(tile_k / mi->block_scaling.block_width,
                                                                  tile_n,
                                                                  B_bs_buf_fast.begin()
                                                                      + (tn * (k / tile_k) + tk) * tile_k * tile_n
                                                                            / mi->block_scaling.block_width,
                                                                  matrix::layout_optimal);
                                D += std::pair(A, A_bs) * std::pair(B, B_bs);
                            }
                            else
                            {
                                D += A * B;
                            }
                        });
                        D.store(Dbuf.begin() + tm * tile_m * n + tn * tile_n, matrix::row_major, n);
                    });
                }
                else
                {
                    const bool use_bulk_copy = device.support_bulk_copy() && USE_BULK_COPY;
                    const bool use_async_copy = device.support_async_copy() && USE_ASYNC_COPY;
                    optional<mbarriers> mbar;
                    if (use_bulk_copy)
                    {
                        mbar.emplace(2, 2 + is_sparse + 2 * with_block_scaling);
                    }

                    Tuint warps_m = exp2(log2(double(ls) / ws) / 2);
                    Tuint warps_n = (ls / ws) / warps_m;
                    Tuint bigtile_m = tile_m * warps_m;
                    Tuint bigtile_n = tile_n * warps_n;
                    assert(m % bigtile_m == 0);
                    assert(n % bigtile_n == 0);
                    assert(k % tile_k == 0);
                    const unsigned int num_slots = (use_bulk_copy || use_async_copy) ? 2 : 1;

                    gpu_for_group(0, (m / bigtile_m) * (n / bigtile_n), [&](gpu_uint bigtile) {
                        gpu_uint btm = bigtile / (n / bigtile_n);
                        gpu_uint btn = bigtile % (n / bigtile_n);
                        gpu_uint tm0 = btm * (bigtile_m / tile_m);
                        gpu_uint tn0 = btn * (bigtile_n / tile_n);
                        gpu_uint warp_id = (local_id() / mi->Nthreads);

                        matrix::warp_matrix<typename make_cpu<c_float_type_nodebug>::type> D(tile_m, tile_n);
                        D.set_info(mi);
                        D.fill(0);

                        array<local_mem<a_float_type>, 2> tmp_a;
                        array<local_mem<b_float_type>, 2> tmp_b;
                        array<local_mem<Tuint>, 2> tmp_sparse_md;
                        array<local_mem<bs_type>, 2> tmp_bs_a;
                        array<local_mem<bs_type>, 2> tmp_bs_b;
                        for (unsigned int slot = 0; slot < num_slots; ++slot)
                        {
                            tmp_a[slot].assign(tile_k * bigtile_m / mi->sparsity(matrix::matrix_a));
                            tmp_b[slot].assign(tile_k * bigtile_n);
                            if constexpr (is_sparse)
                            {
                                tmp_sparse_md[slot].assign(warps_m * sparse_md_tile_size);
                            }
                            if constexpr (with_block_scaling)
                            {
                                tmp_bs_a[slot].assign(tile_k * bigtile_m / mi->block_scaling.block_width);
                                tmp_bs_b[slot].assign(tile_k * bigtile_n / mi->block_scaling.block_width);
                            }
                        }

                        auto fetch = [&](int slot, gpu_uint tk) {
                            auto copy_mem = [&](auto src, unsigned int N, auto dest) {
                                if (use_bulk_copy)
                                {
                                    gpu_if(local_id() == 0)
                                    {
                                        bulk_copy(src, src + N, dest, (*mbar)(slot));
                                    }
                                }
                                else
                                {
                                    parallel_copy(src, N, dest, use_async_copy);
                                }
                            };
                            copy_mem(Abuf_fast.begin()
                                         + (tk * (m / tile_m) + tm0) * tile_k * tile_m / mi->sparsity(matrix::matrix_a),
                                     tile_k * bigtile_m / mi->sparsity(matrix::matrix_a),
                                     tmp_a[slot].begin());
                            if constexpr (is_sparse)
                            {
                                copy_mem(sparse_md.begin() + (tk * (m / tile_m) + tm0) * sparse_md_tile_size,
                                         sparse_md_tile_size * warps_m,
                                         tmp_sparse_md[slot].begin());
                            }
                            copy_mem(Bbuf_fast.begin() + (tk * (n / tile_n) + tn0) * tile_k * tile_n,
                                     tile_k * bigtile_n,
                                     tmp_b[slot].begin());

                            if constexpr (with_block_scaling)
                            {
                                copy_mem(A_bs_buf_fast.begin()
                                             + (tk * (m / tile_m) + tm0) * tile_k * tile_m
                                                   / mi->block_scaling.block_width,
                                         tile_k * bigtile_m / mi->block_scaling.block_width,
                                         tmp_bs_a[slot].begin());
                                copy_mem(B_bs_buf_fast.begin()
                                             + (tk * (n / tile_n) + tn0) * tile_k * tile_n
                                                   / mi->block_scaling.block_width,
                                         tile_k * bigtile_n / mi->block_scaling.block_width,
                                         tmp_bs_b[slot].begin());
                            }
                            if (!use_bulk_copy)
                            {
                                async_commit();
                            }
                        };

                        if (use_bulk_copy || use_async_copy)
                        {
                            // Trigger copying of the first tile.
                            fetch(0, 0);
                        }

                        assert((k / tile_k) % num_slots == 0);
                        gpu_for(0, (k / tile_k), num_slots, [&](gpu_uint tk) {
                            for (unsigned int slot = 0; slot < num_slots; ++slot)
                            {
                                if (use_bulk_copy)
                                {
                                    gpu_if(tk + slot != (k / tile_k) - 1)
                                    {
                                        // Trigger copying of the next tile.
                                        fetch(1 - slot, tk + slot + 1);
                                    }
                                    // And wait for the previous copy operation to finish.
                                    // bulk copy uses mbarrier synchronization mechanism. Alternating between two
                                    // mbarrier objects, with 2 parity states each.
                                    (*mbar)(slot).wait(tk / 2);
                                }
                                else if (use_async_copy)
                                {
                                    gpu_if(tk + slot != (k / tile_k) - 1)
                                    {
                                        // Trigger copying of the next tile.
                                        fetch(1 - slot, tk + slot + 1);

                                        // And wait for the previous copy operation to finish.
                                        async_wait(1);
                                    }
                                    gpu_else
                                    {
                                        // This is the last tile. Need to wait for the data transfer to finish.
                                        async_wait(0);
                                    }
                                    local_barrier(memory::threadgroup);
                                }
                                else
                                {
                                    fetch(0, tk);
                                    local_barrier(memory::threadgroup);
                                }

                                matrix_a_t A(tile_m, tile_k);
                                A.set_info(mi);
                                A.load(tmp_a[slot].begin()
                                           + warp_id / warps_n * tile_k * tile_m / mi->sparsity(matrix::matrix_a),
                                       matrix::layout_optimal);

                                if constexpr (is_sparse)
                                {
                                    A.load_metadata(tmp_sparse_md[slot].begin()
                                                    + warp_id / warps_n * sparse_md_tile_size);
                                }

                                matrix::warp_matrix<typename make_cpu<b_float_type_nodebug>::type> B(
                                    tile_k,
                                    tile_n,
                                    tmp_b[slot].begin() + warp_id % warps_n * tile_k * tile_n,
                                    matrix::layout_optimal);
                                if constexpr (with_block_scaling)
                                {
                                    matrix::warp_matrix<bs_type> A_bs(tile_m,
                                                                      tile_k / mi->block_scaling.block_width,
                                                                      tmp_bs_a[slot].begin()
                                                                          + warp_id / warps_n * tile_k * tile_m
                                                                                / mi->block_scaling.block_width,
                                                                      matrix::layout_optimal);
                                    matrix::warp_matrix<bs_type> B_bs(tile_k / mi->block_scaling.block_width,
                                                                      tile_n,
                                                                      tmp_bs_b[slot].begin()
                                                                          + warp_id % warps_n * tile_k * tile_n
                                                                                / mi->block_scaling.block_width,
                                                                      matrix::layout_optimal);

                                    D += std::pair(A, A_bs) * std::pair(B, B_bs);
                                }
                                else
                                {
                                    D += A * B;
                                }

                                local_barrier(memory::threadgroup);
                            }
                        });
                        D.store(Dbuf.begin() + warp_id / warps_n * tile_m * n + warp_id % warps_n * tile_n
                                    + btm * bigtile_m * n + btn * bigtile_n,
                                matrix::row_major,
                                n);
                    });
                }
            },
            ls,
            0);

        prepare_A.assign(
            device,
            prepare_func<false, a_input_type>(Abuf_fast, nullptr, m, k, tile_m, tile_k, matrix::matrix_a, mi),
            ws,
            0);
        prepare_A_from_dense.assign(
            device,
            prepare_func<true, a_input_type>(Abuf_fast, &sparse_md, m, k, tile_m, tile_k, matrix::matrix_a, mi),
            ws,
            0);
        prepare_B.assign(
            device,
            prepare_func<false, b_input_type>(Bbuf_fast, nullptr, k, n, tile_k, tile_n, matrix::matrix_b, mi),
            ws,
            0);
        if constexpr (with_block_scaling)
        {
            A_bs_buf_fast.assign(device, Abuf_fast.size() / mi->block_scaling.block_width);
            B_bs_buf_fast.assign(device, Bbuf_fast.size() / mi->block_scaling.block_width);

            prepare_A_bs.assign(device,
                                prepare_func<false, bs_type>(A_bs_buf_fast,
                                                             nullptr,
                                                             m,
                                                             k / mi->block_scaling.block_width,
                                                             tile_m,
                                                             tile_k / mi->block_scaling.block_width,
                                                             matrix::matrix_block_scaling_a,
                                                             mi),
                                ws,
                                0);
            prepare_B_bs.assign(device,
                                prepare_func<false, bs_type>(B_bs_buf_fast,
                                                             nullptr,
                                                             k / mi->block_scaling.block_width,
                                                             n,
                                                             tile_k / mi->block_scaling.block_width,
                                                             tile_n,
                                                             matrix::matrix_block_scaling_b,
                                                             mi),
                                ws,
                                0);
        }
    }
};

template<typename a_float_type, typename b_float_type, typename c_float_type, typename scale_type>
Tdouble verify(span<const a_float_type> Adata,
               span<const b_float_type> Bdata,
               span<const c_float_type> Ddata,
               span<const scale_type> A_bs_data,
               span<const scale_type> B_bs_data,
               unsigned int m,
               unsigned int n,
               unsigned int k)
{
    VectorX<Tdouble> test_vector;
    {
        std::default_random_engine generator;
        std::normal_distribution<double> distribution;
        test_vector = VectorX<Tdouble>(N());
        for (Tdouble& e : test_vector)
        {
            e = distribution(generator);
        }
    }
    MatrixX<Tdouble> A =
        Map<Matrix<a_float_type, Dynamic, Dynamic, RowMajor>>(const_cast<a_float_type*>(Adata.data()), m, k)
            .template cast<Tdouble>();
    MatrixX<Tdouble> B =
        Map<Matrix<b_float_type, Dynamic, Dynamic, RowMajor>>(const_cast<b_float_type*>(Bdata.data()), k, n)
            .template cast<Tdouble>();
    MatrixX<Tdouble> D =
        Map<Matrix<c_float_type, Dynamic, Dynamic, RowMajor>>(const_cast<c_float_type*>(Ddata.data()), m, n)
            .template cast<Tdouble>();

    if (VERB)
    {
        cout << "A=\n" << A << endl;
        cout << "B=\n" << B << endl;
        cout << "D=\n" << D << endl;
    }
    if (!A_bs_data.empty())
    {
        unsigned int block_width = Adata.size() / A_bs_data.size();
        MatrixX<scale_type> A_bs = Map<Matrix<scale_type, Dynamic, Dynamic, RowMajor>>(
            const_cast<scale_type*>(A_bs_data.data()), m, k / block_width);
        MatrixX<scale_type> B_bs = Map<Matrix<scale_type, Dynamic, Dynamic, RowMajor>>(
            const_cast<scale_type*>(B_bs_data.data()), k / block_width, n);
        for (unsigned int r = 0; r < m; ++r)
        {
            for (unsigned int c = 0; c < k; ++c)
            {
                A(r, c) *= A_bs(r, c / block_width);
            }
        }
        for (unsigned int r = 0; r < k; ++r)
        {
            for (unsigned int c = 0; c < n; ++c)
            {
                B(r, c) *= B_bs(r / block_width, c);
            }
        }
        if (VERB)
        {
            cout << "scaled_A=\n" << A << endl;
            cout << "scaled_B=\n" << B << endl;
        }
    }
    if (VERB)
    {
        MatrixX<Tdouble> D_cpu = A * B;
        cout << "Dcpu=\n" << D_cpu << endl;
        cout << "diff=\n" << D - D_cpu << endl;
    }

    VectorX<Tdouble> rwant = A * (B * test_vector);
    VectorX<Tdouble> rhave = D * test_vector;

    return (rhave - rwant).norm() / rwant.norm();
}

template<typename a_float_type,
         typename b_float_type,
         typename c_float_type,
         typename a_input_type,
         typename b_input_type,
         bool is_sparse,
         bool with_block_scaling>
void run2(goopax_device device)
{
    Matmul<a_float_type, b_float_type, c_float_type, a_input_type, b_input_type, is_sparse, with_block_scaling> matmul(
        device, M, N, K, TILE_M, TILE_N, TILE_K, LOCAL_SIZE());
    if (!matmul.mi)
    {
        // cout << "  Not supported." << endl;
        return;
    }
    cout << "T_A=" << type_name(type_enum<a_float_type>::value) << ", T_B=" << type_name(type_enum<b_float_type>::value)
         << ", T_C=" << type_name(type_enum<c_float_type>::value) << ", mnk=" << M << "," << N << "," << K
         << ", tile_m=" << matmul.tile_m << ", tile_n=" << matmul.tile_n << ", tile_k=" << matmul.tile_k
         << ", ls=" << matmul.ls << flush;

    buffer<a_input_type> A(device, M * K);
    buffer<b_input_type> B(device, K * N);
    buffer<c_float_type> D(device, M * N);
    fill_random(A);
    fill_random(B);

    buffer<Tdebugtype<precision::fp8ue8m0>> A_bs;
    buffer<Tdebugtype<precision::fp8ue8m0>> B_bs;

    device.wait_all();
    auto t0 = steady_clock::now();

    if (is_sparse)
    {
        make_sparse(A, matmul.mi->sparse_a);
        matmul.prepare_A_from_dense(A);
    }
    else
    {
        matmul.prepare_A(A);
    }
    matmul.prepare_B(B);

    if (with_block_scaling)
    {
        A_bs.assign(device, M * K / matmul.mi->block_scaling.block_width);
        B_bs.assign(device, K * N / matmul.mi->block_scaling.block_width);
        A_bs.fill(2);
        B_bs.fill(4);

        matmul.prepare_A_bs(A_bs);
        matmul.prepare_B_bs(B_bs);
    }

    device.wait_all();
    auto t1 = steady_clock::now();

    matmul.multiply(D);

    device.wait_all();
    auto t2 = steady_clock::now();

    double OPS = double(M) * N * K * 2 / duration_cast<duration<double>>(t2 - t1).count();
    cout << " preparing matrices: " << duration_cast<std::chrono::microseconds>(t1 - t0)
         << ", multiply: " << duration_cast<std::chrono::microseconds>(t2 - t1) << " -> Performance: " << OPS * 1E-12
         << " TOPS" << flush;

    if (VERB)
    {
        if constexpr (!is_same_v<typename remove_debug<a_float_type>::type, precision::tf32>)
        {
            cout << "A_fast=" << matmul.Abuf_fast << endl;
            cout << "B_fast=" << matmul.Bbuf_fast << endl;
        }
        if (with_block_scaling)
        {
            cout << "A_bs_fast=" << matmul.A_bs_buf_fast << endl;
            cout << "B_bs_fast=" << matmul.B_bs_buf_fast << endl;
        }
    }

    // if constexpr (!is_same_v<typename remove_debug<a_float_type>::type, precision::tf32>)
    {
        cout << (is_sparse ? " (sparse)" : "") << (with_block_scaling ? " (block scaling)" : "") << " verifying..."
             << flush;
        Tdouble err =
            verify<a_input_type, b_input_type, c_float_type, Tdebugtype<precision::fp8ue8m0>>(const_buffer_map(A),
                                                                                              const_buffer_map(B),
                                                                                              const_buffer_map(D),
                                                                                              const_buffer_map(A_bs),
                                                                                              const_buffer_map(B_bs),
                                                                                              M,
                                                                                              N,
                                                                                              K);
        cout << " err=" << err << endl;
        assert(err < 0.01f);
    }
}

template<typename a_float_type,
         typename b_float_type,
         typename c_float_type,
         typename a_input_type,
         typename b_input_type>
void run_with_types(goopax_device device)
{
    if constexpr (bitsize<a_float_type>::value >= 8)
    {
        run2<a_float_type, b_float_type, c_float_type, a_input_type, b_input_type, false, false>(device);
        if constexpr (bitsize<a_float_type>::value <= 32)
        {
            run2<a_float_type, b_float_type, c_float_type, a_input_type, b_input_type, true, false>(device);
        }
    }
    run2<a_float_type, b_float_type, c_float_type, a_input_type, b_input_type, false, true>(device);
    if constexpr (bitsize<a_float_type>::value <= 32)
    {
        run2<a_float_type, b_float_type, c_float_type, a_input_type, b_input_type, true, true>(device);
    }
    /*
    run<a_float_type, b_float_type, c_float_type, false>(device, false);
    run<a_float_type, b_float_type, c_float_type, false>(device, true);
    if constexpr (!std::is_same_v<typename remove_debug<a_float_type>::type, double>)
    {
        run<a_float_type, b_float_type, c_float_type, true>(device, false);
        run<a_float_type, b_float_type, c_float_type, true>(device, true);
    }
    */
    // cout << endl;
}

int main(int argc, char** argv)
{
    init_params(argc, argv);

    for (auto device : devices(GOOPAX_DEBUG ? env_CPU : env_GPU))
    {
        cout << "running on device " << device.name() << ", env=" << device.get_envmode() << endl;
        cout << "matrix sizes: matrix<T_A, " << M() << ", " << K() << "> * matrix<T_B, " << K() << ", " << N()
             << "> + matrix<T_C, " << M() << ", " << N() << ">" << endl;

        run_with_types<Thalf, Thalf, Thalf, Thalf, Thalf>(device);
        run_with_types<Thalf, Thalf, Tfloat, Thalf, Thalf>(device);
        run_with_types<Tbfloat16, Tbfloat16, Tfloat, Tbfloat16, Tbfloat16>(device);
        run_with_types<Tfloat, Tfloat, Tfloat, Tfloat, Tfloat>(device);
        run_with_types<Tdebugtype<precision::fp8e4m3>,
                       Tdebugtype<precision::fp8e5m2>,
                       Tfloat,
                       Tdebugtype<precision::fp8e4m3>,
                       Tdebugtype<precision::fp8e5m2>>(device);
        // run_with_types<precision::fp8e3m2, precision::fp8e2m3, Tfloat>(device);
        // run_with_types<precision::fp8e2m1, precision::fp8e2m1, Tfloat>(device);
        run_with_types<Tdebugtype<precision::fp4e2m1>, Tdebugtype<precision::fp4e2m1>, Tfloat, Tfloat, Tfloat>(device);
        run_with_types<Ttf32, Ttf32, Tfloat, Tfloat, Tfloat>(device);
        run_with_types<Tint8_t, Tint8_t, Tint, Tint8_t, Tint8_t>(device);
        run_with_types<Tdebugtype<precision::int4>, Tdebugtype<precision::int4>, Tint, Tint8_t, Tint8_t>(device);
        run_with_types<Tdouble, Tdouble, Tdouble, Tdouble, Tdouble>(device);

        cout << endl;
    }
}
