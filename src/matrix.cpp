/**
   \example matmul.cpp
   matrix multiplication example program, demonstrating the
   use of tensor core hardware acceleration
 */

#include <cassert>
#include <chrono>
#include <goopax>
#include <goopax_draw/types.h>
#include <goopax_extra/param.hpp>
#include <goopax_extra/random.hpp>
#include <goopax_extra/types.hpp>
#include <random>

using namespace Eigen;
using namespace std::chrono;
using namespace goopax;
using namespace std;

// Matrix sizes. Can be specified as command line arguments. See matmul --help
PARAMOPT<unsigned int> M("m", 4096);
PARAMOPT<unsigned int> N("n", 4096);
PARAMOPT<unsigned int> K("k", 4096);

PARAMOPT<unsigned int> BM("bm", 0);
PARAMOPT<unsigned int> BN("bn", 0);
PARAMOPT<unsigned int> BK("bk", 0);

PARAMOPT<bool> COL_MAJOR_A("col_major_a", false);
PARAMOPT<bool> COL_MAJOR_B("col_major_b", true);
PARAMOPT<bool> COL_MAJOR_C("col_major_c", false);

PARAMOPT<bool> REARRANGE("rearrange", true);
PARAMOPT<bool> USE_WORKGROUP_MATRIX("use_workgroup_matrix", true);
PARAMOPT<unsigned int> WORKGROUP_MATRIX_LOCAL_SIZE("workgroup_matrix_local_size", 256);

namespace std
{
template<typename T>
gpu_ostream& operator<<(gpu_ostream& s, const vector<T>& v)
{
    s << "(";
    for (unsigned int k = 0; k < v.size(); ++k)
    {
        if (k != 0)
            s << ", ";
        s << v[k];
    }
    s << ")";
    return s;
}
}

namespace goopax::matrix
{
template<typename T>
struct workgroup_matrix_ab
{
    template<typename P>
    static constexpr bool pointer_valid =
        goopax_is_gpu_pointer_or_sub_byte_pointer<P>::value
        && std::is_same<
            typename std::remove_const<typename goopax_remove_pointer<typename make_cpu<P>::type>::type>::type,
            T>::value;

    unsigned int rows = 0;
    unsigned int cols = 0;
    layout_t layout;

    local_mem<T> storage_i;

    typename make_gpu_pointer<T, memory::threadgroup>::type storage(gpu_uint slot)
    {
        return storage_i.begin() + slot * (rows * cols);
    }
    typename make_gpu_pointer<const T, memory::threadgroup>::type storage(gpu_uint slot) const
    {
        return storage_i.begin() + slot * (rows * cols);
    }

    unsigned int default_pitch(layout_t layout) const
    {
        return (layout == col_major ? rows : cols);
    }

    template<typename P>
        requires pointer_valid<P>
    void load(P ptr, gpu_uint slot, layout_t layout)
    {
        load(ptr, slot, layout, default_pitch(layout));
    }

    template<typename P>
        requires pointer_valid<P>
    void load(P ptr, gpu_uint slot, layout_t layout, gpu_uint pitch)
    {
        this->layout = layout;

        unsigned int rows_use = rows;
        unsigned int cols_use = cols;
        if (layout == col_major)
        {
            swap(rows_use, cols_use);
        }

        using value_type_orig = typename goopax_remove_pointer<typename make_cpu<P>::type>::type;
        using value_type_use =
            typename std::conditional<(get_bits<value_type_orig>::value < 32), int, value_type_orig>::type;

        using P_use_src =
            typename make_gpu_pointer<value_type_use, get_pointer_scope<typename make_cpu<P>::type>::value>::type;
        using P_use_dest = typename make_gpu_pointer<value_type_use, memory::threadgroup>::type;

        P_use_src ptr_use = reinterpret<P_use_src>(ptr);
        constexpr unsigned int size_factor = get_bits<value_type_use>::value / get_bits<value_type_orig>::value;
        pitch /= size_factor;
        cols_use /= size_factor;

        P_use_dest ptr_dest = reinterpret<P_use_dest>(storage(slot));

        if (false)
        {
            gpu_for_local(0,
                          rows_use * cols_use,
                          par_unroll(std::min(rows * cols / local_size(),
                                              static_cast<unsigned int>(16 * 8 / get_bits<value_type_use>::value))),
                          [&](gpu_uint k) { ptr_dest[k] = ptr[k + k / cols_use * (pitch - cols_use)]; });
        }
        else
        {
            unsigned int step = std::min(rows_use * cols_use / local_size(),
                                         static_cast<unsigned int>(16 * 8 / get_bits<value_type_use>::value));

            unsigned int k;
            for (k = 0; k < rows_use * cols_use; k += step * local_size())
            {
                for (unsigned int m = 0; m < step; ++m)
                {
                    ptr_dest[k + local_id() * step + m] = ptr_use[k + local_id() * step + m];
                }
            }
            assert(k == rows_use * cols_use);
        }
    }

    template<typename P>
        requires pointer_valid<P>
    void load_async(P ptr, gpu_uint slot, layout_t layout)
    {
        load_async(ptr, slot, layout, default_pitch(layout));
    }

    template<typename P>
        requires pointer_valid<P>
    void load_async(P ptr, gpu_uint slot, layout_t layout, gpu_uint pitch)
    {
        this->layout = layout;

        unsigned int rows_use = rows;
        unsigned int cols_use = cols;
        if (layout == col_major)
        {
            swap(rows_use, cols_use);
        }

        using value_type_orig = typename goopax_remove_pointer<typename make_cpu<P>::type>::type;
        using value_type_use =
            typename std::conditional<(get_bits<value_type_orig>::value < 32), int, value_type_orig>::type;

        using P_use_src =
            typename make_gpu_pointer<value_type_use, get_pointer_scope<typename make_cpu<P>::type>::value>::type;
        using P_use_dest = typename make_gpu_pointer<value_type_use, memory::threadgroup>::type;

        P_use_src ptr_use = reinterpret<P_use_src>(ptr);
        constexpr unsigned int size_factor = get_bits<value_type_use>::value / get_bits<value_type_orig>::value;
        pitch /= size_factor;
        cols_use /= size_factor;

        P_use_dest ptr_dest = reinterpret<P_use_dest>(storage(slot));

        // using value_type = typename goopax_remove_pointer<typename make_cpu<P>::type>::type;

        if (false)
        {
            gpu_for_local(0,
                          rows_use * cols_use,
                          par_unroll(std::min(rows_use * cols_use / local_size(),
                                              static_cast<unsigned int>(16 * 8 / get_bits<value_type_use>::value))),
                          [&](gpu_uint k) {
                              async_copy(ptr_use + k + k / cols_use * (pitch - cols_use), ptr_dest + k);
                              // storage[k] = ptr_use[k + k / cols_use * (pitch - cols_use)];
                          });
        }
        else
        {
            unsigned int step = std::min(rows_use * cols_use / local_size(),
                                         static_cast<unsigned int>(16 * 8 / get_bits<value_type_use>::value));

            unsigned int k;
            for (k = 0; k < rows_use * cols_use; k += step * local_size())
            {
                for (unsigned int m = 0; m < step; ++m)
                {
                    async_copy(ptr_use + k + local_id() * step + m, ptr_dest + k + local_id() * step + m);
                }
            }
            assert(k == rows_use * cols_use);
        }
    }

    template<typename P>
        requires pointer_valid<P>
    workgroup_matrix_ab(unsigned int rows, unsigned int cols, P ptr, layout_t layout, gpu_int pitch)
        : workgroup_matrix_ab(rows, cols)
    {
        load(ptr, layout, pitch);
    }

    workgroup_matrix_ab(unsigned int rows0, unsigned int cols0, unsigned int num_slots)
        : rows(rows0)
        , cols(cols0)
        , storage_i(rows * cols * num_slots)
    {
    }
};

template<typename T>
struct workgroup_matrix_c
{
    unsigned int rows = 0;
    unsigned int cols = 0;
    unsigned int brows;
    unsigned int bcols;

    warp_matrix<T> tile;

    template<typename P>
    static constexpr bool pointer_valid =
        goopax_is_gpu_pointer_or_sub_byte_pointer<P>::value
        && std::is_same<
            typename std::remove_const<typename goopax_remove_pointer<typename make_cpu<P>::type>::type>::type,
            T>::value;

    /**
       Operator+=
       Adds the matrix product `ab` to the matrix.
    */
    /*
      template<typename T_A, typename T_B>
      workgroup_matrix_c& operator+=(const workgroup_matrix_product<T_A, T_B>& ab)
      {
          gpu_uint br = warp_id_in_group() / bcols;
          gpu_uint bc = warp_id_in_group() % bcols;
          // gpu_cout << "thread=" << global_id() << ". loading a from storage.begin() + " << br*(rows/brows)*ab.a.cols
          //<< " and b from storage.begin() + " << bc*(cols/bcols)*ab.a.rows << "\n";
          warp_matrix<T_A> a(rows / brows, ab.a.cols, ab.a.storage.begin() + br * (rows / brows) * ab.a.cols,
      row_major); warp_matrix<T_B> b(ab.a.cols, cols / bcols, ab.b.storage.begin() + bc * (cols / bcols) * ab.b.rows,
      col_major); multiply_add(a, b, tile);
          // gpu_cout << "  operator+=. before: tile=" << tile.coeffs();
          tile += a * b;
          // gpu_cout << "\n  a=" << a.coeffs()
          //<< "\n  b=" << b.coeffs()
          //<< "\n  -> tile=" << tile.coeffs() << "\n";
          return *this;
      }
    */
    template<typename T_A, typename T_B>
    void add_product(const workgroup_matrix_ab<T_A>& wa, const workgroup_matrix_ab<T_B>& wb, gpu_uint slot)
    {
        gpu_uint br = warp_id_in_group() / bcols;
        gpu_uint bc = warp_id_in_group() % bcols;
        warp_matrix<T_A> a(rows / brows, wa.cols, wa.storage(slot) + br * (rows / brows) * wa.cols, row_major);
        warp_matrix<T_B> b(wa.cols, cols / bcols, wb.storage(slot) + bc * (cols / bcols) * wb.rows, col_major);
        // multiply_add(a, b, tile);
        tile += a * b;
    }

    void fill(const typename make_gpu<T>::type& value)
    {
        tile.fill(value);
    }

    template<typename P>
        requires(pointer_valid<P> && !std::is_const<typename goopax_remove_pointer<P>::type>::value)
    void store(P ptr, layout_t layout, gpu_uint pitch)
    {
        tile.store(
            ptr
                + (layout == row_major
                       ? warp_id_in_group() / bcols * tile.rows * pitch + warp_id_in_group() % bcols * (cols / bcols)
                       : warp_id_in_group() % bcols * pitch + warp_id_in_group() / bcols * tile.rows),
            layout,
            pitch);
    }

    workgroup_matrix_c(unsigned int rows0, unsigned int cols0)
        : rows(rows0)
        , cols(cols0)
    {
        brows = 1;
        while (brows * 2 * brows * 2 <= local_size() / warp_size())
        {
            brows *= 2;
        }
        bcols = local_size() / warp_size() / brows;
        tile = warp_matrix<T>(rows / brows, cols / bcols);
        cout << "workgroup_matrix_c: rows=" << rows << ", cols=" << cols << ", brows=" << brows << ", bcols=" << bcols
             << endl;
        cout << "local_size()=" << local_size() << endl;
        cout << "global_size()=" << global_size() << endl;
    }
};

/**
   Temporary expression for a matrix product.
   \ingroup warp_matrix
*/
/*template<typename T_A, typename T_B>
struct workgroup_matrix_product
{
    const workgroup_matrix_ab<T_A>& a;
    const workgroup_matrix_ab<T_B>& b;
  const gpu_uint slot;
};
*/

/**
   operator*
   \ingroup warp_matrix
   Returns a temporary expression. The result can be used with operator+ or operator+=.
 */
/*
  template<typename T_A, typename T_B>
workgroup_matrix_product<T_A, T_B> operator*(const workgroup_matrix_ab<T_A>& a, const workgroup_matrix_ab<T_B>& b)
{
    return workgroup_matrix_product<T_A, T_B>{ a, b };
}
*/

}

template<typename a_float_type, typename c_float_type>
void fill_random(WELL512_data& rnd,
                 buffer<a_float_type>& a,
                 buffer<c_float_type>& ad,
                 unsigned int brows,
                 unsigned int bcols,
                 unsigned int cols)
{
    kernel fill_random(
        a.get_device(),
        [&rnd](resource<a_float_type>& a, resource<c_float_type>& ad, gpu_uint brows, gpu_uint bcols, gpu_uint cols) {
            WELL512_lib rndlib(rnd);

            // Parallelizing memory access such that it is compatible with sub_byte_pointer access for int4.
            unsigned int par = max(32u / static_cast<unsigned int>(get_bits<a_float_type>::value), 1u);
            gpu_for_global(0, a.size(), par, [&](gpu_uint k) {
                auto pd = ad.begin() + k;
                auto p = a.begin() + k;

                if (REARRANGE)
                {
                    gpu_uint row = k / cols;
                    gpu_uint col = k % cols;
                    p = a.begin() + col / bcols * brows * bcols + col % bcols + row % brows * bcols
                        + row / brows * brows * cols;
                }

                for (unsigned int sub = 0; sub < par; ++sub)
                {
                    if constexpr (std::is_same_v<typename make_cpu<a_float_type>::type, precision::tf32>)
                    {
                        gpu_float v = rndlib.gaussian_distribution();
                        pd[sub] = v;
                        p[sub] = static_cast<typename make_gpu<a_float_type>::type>(v);
                    }
                    else
                    {
                        pd[sub] = p[sub] =
                            static_cast<typename make_gpu<a_float_type>::type>(rndlib.gaussian_distribution());
                    }
                }
            });
        });
    fill_random(a, ad, brows, bcols, cols);
}

template<typename a_float_type, typename b_float_type, typename c_float_type>
void run_with_types(goopax_device device)
try
{
    cout << "\n\nUsing types T_A=" << type_name(type_enum<a_float_type>::value)
         << ", T_B=" << type_name(type_enum<b_float_type>::value)
         << " and T_C=" << type_name(type_enum<c_float_type>::value) << endl;

    // Choosing suitable matrix block sizes.
    // Larger values can improve performance, but only if there are
    // enough registers available.
    unsigned int bm;
    unsigned int bn;
    unsigned int bk;
    if (USE_WORKGROUP_MATRIX)
    {
        assert(REARRANGE);
        bm = 256;
        bn = 256;
        bk = 16;
    }
    else
    {
        bm = 64;
        bn = 64;
        bk = 16;
        if (device.max_registers() < 80)
        {
            bm = 32;
            bn = 32;
        }
        if (get_bits<a_float_type>::value == 8)
        {
            bk = 64;
        }
        if (get_bits<a_float_type>::value == 4)
        {
            if (goopax_is_integral<a_float_type>::value)
            {
                bk = 64;
                bm = 64;
                bn = 32;
            }
            else
            {
                bk = 64;
                bm = 64;
                bn = 64;
            }
        }
    }
    if (BM())
    {
        bm = BM();
    }
    if (BN())
    {
        bn = BN();
    }
    if (BK())
    {
        bk = BK();
    }

    assert(M % bm == 0);
    assert(N % bn == 0);
    assert(K % bk == 0);

    const bool use_bulk_copy = device.support_bulk_copy();

    unsigned int Nthreads = device.default_local_size();

    // Matrix buffers
    buffer<a_float_type> A(device, M * K);
    buffer<b_float_type> B(device, K * N);
    buffer<c_float_type> C(device, M * N);

    // Matrix buffers for verification
    buffer<c_float_type> Ad(device, M * K);
    buffer<c_float_type> Bd(device, K * N);

    cout << "Memory requirements [MB]: " << (A.size() * get_bits<a_float_type>::value / 8 >> 16) << " + "
         << (B.size() * get_bits<b_float_type>::value / 8 >> 16) << " + " << (C.size() * sizeof(c_float_type) >> 16)
         << " = "
         << ((A.size() * get_bits<a_float_type>::value / 8 + B.size() * get_bits<b_float_type>::value / 8
              + C.size() * sizeof(c_float_type))
             >> 16)
         << ", device cache: ";
    if (device.cache_size() == 0)
        cout << "unknown";
    else
        cout << (device.cache_size() >> 16);
    cout << "\nRegisters required: "
         << bm * bk * get_bits<a_float_type>::value / Nthreads / 32
                + bk * bn * get_bits<b_float_type>::value / Nthreads / 32
                + bm * bn * get_bits<c_float_type>::value / Nthreads / 32
         << " / " << device.max_registers() << endl;

    // Filling with random numbers
    std::random_device rd;
    WELL512_data rnd(device, device.default_global_size_max(), rd());

    fill_random(rnd, A, Ad, bm, bk, K);
    fill_random(rnd, B, Bd, bn, bk, K);
    C.fill(numeric_limits<c_float_type>::quiet_NaN());

    // Creating the kernel
    kernel<void(buffer<a_float_type> & A, buffer<b_float_type> & B, buffer<c_float_type> & C)> multiply;

    if (USE_WORKGROUP_MATRIX)
    {
        multiply.assign(
            device,
            [bm, bn, bk, use_bulk_copy](
                resource<a_float_type>& A, resource<b_float_type>& B, resource<c_float_type>& C) {
                mbarriers mbar(2, 2);

                gpu_for_group(0, (M / bm) * (N / bn), [&](gpu_uint block) {
                    gpu_uint block_m = block / (N / bn);
                    gpu_uint block_n = block % (N / bn);

                    gpu_uint moff = block_m * bm;
                    gpu_uint noff = block_n * bn;

                    matrix::workgroup_matrix_c<c_float_type> mc(bm, bn);
                    mc.fill(static_cast<c_float_type>(0));

                    matrix::workgroup_matrix_ab<a_float_type> ma(bm, bk, 2);
                    matrix::workgroup_matrix_ab<b_float_type> mb(bk, bn, 2);

                    auto load_data = [&](gpu_uint block_k) {
                        gpu_uint koff = block_k * bk;

                        if (use_bulk_copy)
                        {
                            gpu_if(local_id() == 0)
                            {
                                // gpu_if(elect().first)
                                //{

                                bulk_copy(
                                    B.begin() + (COL_MAJOR_B ? koff * bn + noff * K() : koff * N() + noff * bk),
                                    B.begin()
                                        + ((COL_MAJOR_B ? koff * bn + noff * K() : koff * N() + noff * bk) + bk * bn),
                                    mb.storage(block_k % 2),
                                    mbar(block_k % 2));
                                bulk_copy(
                                    A.begin() + (COL_MAJOR_A ? moff * bk + koff * M() : moff * K() + koff * bm),
                                    A.begin()
                                        + (COL_MAJOR_A ? moff * bk + koff * M() : moff * K() + koff * bm + bm * bk),
                                    ma.storage(block_k % 2),
                                    mbar(block_k % 2));
                            }
                            // mbar.wait(count % 2);

                            ma.layout = COL_MAJOR_A() ? matrix::col_major : matrix::row_major;
                            mb.layout = COL_MAJOR_B() ? matrix::col_major : matrix::row_major;
                            //++count;
                        }
                        else
                        {
                            if (REARRANGE)
                            {
                                ma.load_async(A.begin()
                                                  + (COL_MAJOR_A ? moff * bk + koff * M() : moff * K() + koff * bm),
                                              block_k % 2,
                                              COL_MAJOR_A() ? matrix::col_major : matrix::row_major);
                                mb.load_async(B.begin()
                                                  + (COL_MAJOR_B ? koff * bn + noff * K() : koff * N() + noff * bk),
                                              block_k % 2,
                                              COL_MAJOR_B() ? matrix::col_major : matrix::row_major);
                            }
                            else
                            {
                                ma.load_async(A.begin() + (COL_MAJOR_A ? moff + koff * M() : moff * K() + koff),
                                              block_k % 2,
                                              COL_MAJOR_A() ? matrix::col_major : matrix::row_major,
                                              COL_MAJOR_A() ? M() : K());
                                mb.load_async(B.begin() + (COL_MAJOR_B ? koff + noff * K() : koff * N() + noff),
                                              block_k % 2,
                                              COL_MAJOR_B() ? matrix::col_major : matrix::row_major,
                                              COL_MAJOR_B() ? K() : N());
                            }

                            async_commit();
                        }
                    };

                    load_data(0);

                    gpu_for(0, (K / bk), [&](gpu_uint block_k) {
                        gpu_if(block_k != (K / bk) - 1)
                        {
                            load_data(block_k + 1);
                            if (!use_bulk_copy)
                            {
                                async_wait(1);
                            }
                        }
                        gpu_else
                        {
                            if (!use_bulk_copy)
                            {
                                async_wait(0);
                            }
                        }
                        if (use_bulk_copy)
                        {
                            mbar(block_k % 2).wait(block_k / 2);
                        }
                        else
                        {
                            local_barrier(memory::threadgroup);
                        }

                        // Multiplying matrix tiles, adding the result.
                        mc.add_product(ma, mb, block_k % 2);

                        local_barrier(memory::threadgroup);
                    });

                    mc.store(C.begin() + (COL_MAJOR_C ? moff + noff * M() : moff * N() + noff),
                             COL_MAJOR_C() ? matrix::col_major : matrix::row_major,
                             COL_MAJOR_C() ? M() : N());
                });
            },
            WORKGROUP_MATRIX_LOCAL_SIZE(),
            0);
    }
    else
    {
        multiply.assign(
            device, [bm, bn, bk](resource<a_float_type>& A, resource<b_float_type>& B, resource<c_float_type>& C) {
                gpu_for_group(0, (M / bm) * (N / bn), [&](gpu_uint block) {
                    gpu_uint block_m = block / (N / bn);
                    gpu_uint block_n = block % (N / bn);

                    gpu_uint moff = block_m * bm;
                    gpu_uint noff = block_n * bn;

                    matrix::warp_matrix<c_float_type> mc(bm, bn);
                    mc.fill(static_cast<c_float_type>(0));

                    gpu_for(0, K(), bk, [&](gpu_uint koff) {
                        // Loading matrix tile of Matrix A.
                        matrix::warp_matrix<a_float_type> ma(bm, bk);
                        matrix::warp_matrix<b_float_type> mb(bk, bn);

                        if (REARRANGE)
                        {
                            ma.load(A.begin() + (COL_MAJOR_A ? moff * bk + koff * M() : moff * K() + koff * bm),
                                    COL_MAJOR_A() ? matrix::col_major : matrix::row_major);

                            // Loading matrix tile of Matrix B.
                            mb.load(B.begin() + (COL_MAJOR_B ? koff * bn + noff * K() : koff * N() + noff * bk),
                                    COL_MAJOR_B() ? matrix::col_major : matrix::row_major);
                        }
                        else
                        {
                            ma.load(A.begin() + (COL_MAJOR_A ? moff + koff * M() : moff * K() + koff),
                                    COL_MAJOR_A() ? matrix::col_major : matrix::row_major,
                                    COL_MAJOR_A() ? M() : K());

                            // Loading matrix tile of Matrix B.
                            mb.load(B.begin() + (COL_MAJOR_B ? koff + noff * K() : koff * N() + noff),
                                    COL_MAJOR_B() ? matrix::col_major : matrix::row_major,
                                    COL_MAJOR_B() ? K() : N());
                        }

                        // Multiplying matrix tiles, adding the result.
                        if constexpr (is_same_v<a_float_type, precision::fp4e2m1>)
                        {
                            matrix::warp_matrix<precision::fp8ue8m0> scale_a(ma.rows, ma.cols / 32);
                            matrix::warp_matrix<precision::fp8ue8m0> scale_b(mb.rows / 32, mb.cols);
                            scale_a.fill(static_cast<precision::fp8ue8m0>(1));
                            scale_b.fill(static_cast<precision::fp8ue8m0>(1));
                            mc = multiply_add(ma, mb, mc, scale_a, scale_b);
                        }
                        else
                        {
                            mc += ma * mb;
                        }
                    });

                    mc.store(C.begin() + (COL_MAJOR_C ? moff + noff * M() : moff * N() + noff),
                             COL_MAJOR_C() ? matrix::col_major : matrix::row_major,
                             COL_MAJOR_C() ? M() : N());
                });
            });
    }

    for (unsigned int count = 0; count < 3; ++count)
    {
        device.wait_all();

        auto time_start = steady_clock::now();
        multiply(A, B, C).wait();
        auto time_end = steady_clock::now();

        Tdouble time = duration_cast<duration<double>>(time_end - time_start).count();
        auto OPS = Tdouble(M()) * K() * N() * 2 / time;
        cout << "Did matrix multiplication in " << time << " seconds. Performance: " << OPS / 1E12 << " TOPS" << endl;
    }
    cout << "verifying... " << flush;

    VectorX<double> test_vector;
    {
        std::default_random_engine generator;
        std::normal_distribution<double> distribution;
        test_vector = VectorX<double>(N());
        for (double& e : test_vector)
        {
            e = distribution(generator);
        }
    }

    auto get_matrix = [](buffer_map<c_float_type> M, bool col_major, int rows, int cols) -> MatrixX<double> {
        if (col_major)
        {
            return Map<Matrix<c_float_type, Dynamic, Dynamic, ColMajor>>(M.data(), rows, cols).template cast<double>();
        }
        else
        {
            return Map<Matrix<c_float_type, Dynamic, Dynamic, RowMajor>>(M.data(), rows, cols).template cast<double>();
        }
    };

    MatrixX<double> TA = get_matrix(Ad, COL_MAJOR_A(), M, K);
    MatrixX<double> TB = get_matrix(Bd, COL_MAJOR_B(), K, N);
    MatrixX<double> TC = get_matrix(C, COL_MAJOR_C(), M, N);

    /*
      cout << "A=\n" << TA << endl;
    cout << "B=\n" << TB << endl;
    cout << "C=\n" << TC << endl;
    cout << "C_CPU=\n" << (TA * TB) << endl;
    cout << "\ndiff=\n" << TC - (TA * TB) << endl;
    */

    VectorX<double> rwant = TA * (TB * test_vector);
    VectorX<double> rhave = TC.template cast<double>() * test_vector;

    cout << "err=" << (rhave - rwant).norm() / rwant.norm() << endl;
}
catch (EX::unsupported& e)
{
    cout << "Got exception '" << e.what() << "'" << endl;
}

int main(int argc, char** argv)
{
    init_params(argc, argv);

    for (auto device : devices(GOOPAX_DEBUG ? env_CPU : env_GPU))
    {
        cout << "running on device " << device.name() << ", env=" << device.get_envmode() << endl;
        cout << "matrix sizes: matrix<T_A, " << M() << ", " << K() << "> * matrix<T_B, " << K() << ", " << N()
             << "> + matrix<T_C, " << M() << ", " << N() << ">" << endl;

        run_with_types<Thalf, Thalf, Thalf>(device);
        run_with_types<Thalf, Thalf, Tfloat>(device);
        run_with_types<Tbfloat16, Tbfloat16, Tfloat>(device);
        // run_with_types<Tfloat, Tfloat, Tfloat>(device);
        run_with_types<precision::fp8e4m3, precision::fp8e5m2, Tfloat>(device);
        run_with_types<precision::fp8e3m2, precision::fp8e2m3, Tfloat>(device);
        run_with_types<precision::fp8e2m1, precision::fp8e2m1, Tfloat>(device);
        run_with_types<precision::fp4e2m1, precision::fp4e2m1, Tfloat>(device);
        run_with_types<Ttf32, Ttf32, Tfloat>(device);
        run_with_types<Tint8_t, Tint8_t, Tint>(device);
        run_with_types<precision::int4, precision::int4, Tint>(device);
        run_with_types<Tdouble, Tdouble, Tdouble>(device);

        cout << endl << endl;
    }
}
