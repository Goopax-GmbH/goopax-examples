#pragma once
#include <goopax>

namespace goopax::matrix
{

inline unsigned int warpgroup_rows(unsigned int ls, unsigned int Nthreads)
{
    unsigned int brows = 1;
    while (brows * 2 * brows * 2 <= ls / Nthreads)
    {
        brows *= 2;
    }
    return brows;
}
inline unsigned int warpgroup_cols(unsigned int ls, unsigned int Nthreads)
{
    return ls / Nthreads / warpgroup_rows(ls, Nthreads);
}

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
    const matrix_support_info* mi = nullptr;

    unsigned int num_slots = 0;
    local_mem<T> storage_i;
    unsigned int sparse_md_size = 0;
    local_mem<unsigned int> sparse_md_i;
    unsigned int sparsity() const
    {
        return (mi ? mi->sparse_a.sparsity() : 1u);
    }

    typename make_gpu_pointer<T, memory::threadgroup>::type storage(gpu_uint slot)
    {
        return storage_i.begin() + slot * (rows * cols / sparsity());
    }
    typename make_gpu_pointer<const T, memory::threadgroup>::type storage(gpu_uint slot) const
    {
        return storage_i.begin() + slot * (rows * cols / sparsity());
    }
    auto sparse_md(gpu_uint slot)
    {
        return sparse_md_i.begin() + slot * (sparse_md_size);
    }
    auto sparse_md(gpu_uint slot) const
    {
        return sparse_md_i.begin() + slot * (sparse_md_size);
    }

    unsigned int default_pitch(layout_t layout) const
    {
        return (layout == col_major ? rows : cols) / sparsity();
    }

    template<typename P, typename = typename std::enable_if<pointer_valid<P>>::type>
    void load(P ptr, gpu_uint slot, layout_t layout)
    {
        load(ptr, slot, layout, default_pitch(layout));
    }

    template<typename P, typename = typename std::enable_if<pointer_valid<P>>::type>
    void load(P ptr, gpu_uint slot, layout_t layout, gpu_uint pitch)
    {
        this->layout = layout;

        unsigned int rows_use = rows;
        unsigned int cols_use = cols;
        if (layout == col_major)
        {
            std::swap(rows_use, cols_use);
        }

        using value_type_orig = typename goopax_remove_pointer<typename make_cpu<P>::type>::type;
        using value_type_use =
            typename std::conditional<(bitsize<value_type_orig>::value < 32), int, value_type_orig>::type;

        using P_use_src =
            typename make_gpu_pointer<value_type_use, get_pointer_scope<typename make_cpu<P>::type>::value>::type;
        using P_use_dest = typename make_gpu_pointer<value_type_use, memory::threadgroup>::type;

        P_use_src ptr_use = reinterpret<P_use_src>(ptr);
        const unsigned int size_factor = bitsize<value_type_use>::value / bitsize<value_type_orig>::value * sparsity();
        pitch /= size_factor;
        cols_use /= size_factor;

        P_use_dest ptr_dest = reinterpret<P_use_dest>(storage(slot));

        if (false)
        {
            gpu_for_local(0,
                          rows_use * cols_use,
                          par_unroll(std::min(rows * cols / local_size(),
                                              static_cast<unsigned int>(16 * 8 / bitsize<value_type_use>::value))),
                          [&](gpu_uint k) { ptr_dest[k] = ptr[k + k / cols_use * (pitch - cols_use)]; });
        }
        else
        {
            unsigned int step = std::min(rows_use * cols_use / local_size(),
                                         static_cast<unsigned int>(16 * 8 / bitsize<value_type_use>::value));

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

    template<typename P, typename = typename std::enable_if<pointer_valid<P>>::type>
    void load_async(P ptr, gpu_uint slot, layout_t layout)
    {
        load_async(ptr, slot, layout, default_pitch(layout));
    }

    template<typename P, typename = typename std::enable_if<pointer_valid<P>>::type>
    void load_async(P ptr, gpu_uint slot, layout_t layout, gpu_uint pitch)
    {
        this->layout = layout;

        unsigned int rows_use = rows;
        unsigned int cols_use = cols;
        if (layout == col_major)
        {
            std::swap(rows_use, cols_use);
        }

        using value_type_orig = typename goopax_remove_pointer<typename make_cpu<P>::type>::type;
        using value_type_use =
            typename std::conditional<(bitsize<value_type_orig>::value < 32), int, value_type_orig>::type;

        using P_use_src =
            typename make_gpu_pointer<value_type_use, get_pointer_scope<typename make_cpu<P>::type>::value>::type;
        using P_use_dest = typename make_gpu_pointer<value_type_use, memory::threadgroup>::type;

        P_use_src ptr_use = reinterpret<P_use_src>(ptr);
        const unsigned int size_factor = bitsize<value_type_use>::value / bitsize<value_type_orig>::value * sparsity();
        pitch /= size_factor;
        cols_use /= size_factor;

        P_use_dest ptr_dest = reinterpret<P_use_dest>(storage(slot));

        if (false)
        {
            gpu_for_local(
                0,
                rows_use * cols_use,
                par_unroll(std::min(rows_use * cols_use / local_size(),
                                    static_cast<unsigned int>(16 * 8 / bitsize<value_type_use>::value))),
                [&](gpu_uint k) { async_copy(ptr_use + k + k / cols_use * (pitch - cols_use), ptr_dest + k); });
        }
        else
        {
            unsigned int step = std::min(rows_use * cols_use / local_size(),
                                         static_cast<unsigned int>(16 * 8 / bitsize<value_type_use>::value));

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

    template<typename P, typename = typename std::enable_if<pointer_valid<P>>::type>
    workgroup_matrix_ab(unsigned int rows,
                        unsigned int cols,
                        P ptr,
                        layout_t layout,
                        gpu_int pitch,
                        const matrix_support_info* mi0 = nullptr)
        : workgroup_matrix_ab(rows, cols)
    {
        this->mi = mi0;
        load(ptr, layout, pitch);
    }

    workgroup_matrix_ab(unsigned int rows0,
                        unsigned int cols0,
                        unsigned int num_slots0,
                        const matrix_support_info* mi0 = nullptr)
        : rows(rows0)
        , cols(cols0)
        , mi(mi0)
        , num_slots(num_slots0)
        , storage_i(rows * cols * num_slots)
    {
        if (mi)
        {
            sparse_matrix<T> probe(rows / warpgroup_rows(local_size(), mi->Nthreads), cols, *mi);
            sparse_md_size = probe.sparse_metadata().size() * mi->Nthreads * warpgroup_rows(local_size(), mi->Nthreads);
            sparse_md_i.assign(sparse_md_size * num_slots);
        }
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

    template<typename T_A, typename T_B>
    void add_product(const workgroup_matrix_ab<T_A>& wa, const workgroup_matrix_ab<T_B>& wb, gpu_uint slot)
    {
        gpu_uint br = warp_id_in_group() / bcols;
        gpu_uint bc = warp_id_in_group() % bcols;
        warp_matrix<T_B> b;
        if (wb.layout == col_major)
        {
            b = warp_matrix<T_B>(wa.cols, cols / bcols, wb.storage(slot) + bc * (cols / bcols) * wb.rows, wb.layout);
        }
        else
        {
            b = warp_matrix<T_B>(wa.cols, cols / bcols, wb.storage(slot) + bc * (cols / bcols), wb.layout, wb.cols);
        }
        if (wa.sparsity() != 1)
        {
            assert(wa.layout == row_major);

            using namespace std;
            sparse_matrix<T_A> a(rows / brows, wa.cols, *wa.mi);
            a.load(wa.storage(slot) + br * ((rows / brows) * wa.cols / wa.sparsity()), matrix::row_major);
            a.load_metadata(wa.sparse_md(slot)
                            + br * static_cast<unsigned int>(a.sparse_metadata().size()) * wa.mi->Nthreads);
            tile = multiply_add(a, b, tile);
        }
        else
        {
            warp_matrix<T_A> a;
            if (wa.layout == row_major)
            {
                a = warp_matrix<T_A>(
                    rows / brows, wa.cols, wa.storage(slot) + br * (rows / brows) * wa.cols, wa.layout);
            }
            else
            {
                a = warp_matrix<T_A>(rows / brows, wa.cols, wa.storage(slot) + br * (rows / brows), wa.layout, wa.rows);
            }
            tile += a * b;
        }
    }

    template<typename T_A, typename T_B, typename T_block>
    void add_product(const workgroup_matrix_ab<T_A>& wa,
                     const workgroup_matrix_ab<T_B>& wb,
                     gpu_uint slot,
                     const warp_matrix<T_block>& block_scaling_a,
                     const warp_matrix<T_block>& block_scaling_b)
    {
        gpu_uint br = warp_id_in_group() / bcols;
        gpu_uint bc = warp_id_in_group() % bcols;
        warp_matrix<T_B> b;
        if (wb.layout == col_major)
        {
            b = warp_matrix<T_B>(wa.cols, cols / bcols, wb.storage(slot) + bc * (cols / bcols) * wb.rows, wb.layout);
        }
        else
        {
            b = warp_matrix<T_B>(wa.cols, cols / bcols, wb.storage(slot) + bc * (cols / bcols), wb.layout, wb.cols);
        }
        if (wa.sparsity() != 1)
        {
            assert(wa.layout == row_major);

            using namespace std;
            sparse_matrix<T_A> a(rows / brows, wa.cols, *wa.mi);
            a.load(wa.storage(slot) + br * ((rows / brows) * wa.cols / wa.sparsity()), matrix::row_major);
            a.load_metadata(wa.sparse_md(slot)
                            + br * static_cast<unsigned int>(a.sparse_metadata().size()) * wa.mi->Nthreads);
            tile = multiply_add(a, b, tile, block_scaling_a, block_scaling_b);
        }
        else
        {
            warp_matrix<T_A> a;
            if (wa.layout == row_major)
            {
                a = warp_matrix<T_A>(
                    rows / brows, wa.cols, wa.storage(slot) + br * (rows / brows) * wa.cols, wa.layout);
            }
            else
            {
                a = warp_matrix<T_A>(rows / brows, wa.cols, wa.storage(slot) + br * (rows / brows), wa.layout, wa.rows);
            }
            tile = multiply_add(a, b, tile, block_scaling_a, block_scaling_b);
        }
    }

    void fill(const typename make_gpu<T>::type& value)
    {
        tile.fill(value);
    }

    template<typename P,
             typename = typename std::enable_if<
                 pointer_valid<P> && !std::is_const<typename goopax_remove_pointer<P>::type>::value>::type>
    void store(P ptr, layout_t layout, gpu_uint pitch)
    {
        tile.store(
            ptr
                + (layout == row_major
                       ? warp_id_in_group() / bcols * tile.rows * pitch + warp_id_in_group() % bcols * (cols / bcols)
                       : warp_id_in_group() / bcols * tile.rows + warp_id_in_group() % bcols * (cols / bcols) * pitch),
            layout,
            pitch);
    }

    workgroup_matrix_c(unsigned int rows0, unsigned int cols0)
        : rows(rows0)
        , cols(cols0)
        , brows(warpgroup_rows(local_size(), warp_size()))
        , bcols(warpgroup_cols(local_size(), warp_size()))
    {
        tile = warp_matrix<T>(rows / brows, cols / bcols);
    }
};

template<typename a_float_type, typename b_float_type, typename c_float_type>
void create_matmul_kernel_common(resource<a_float_type>& A,
                                 resource<b_float_type>& B,
                                 resource<c_float_type>& C,
                                 resource<unsigned int>* Asp_md,
                                 unsigned int M,
                                 unsigned int N,
                                 unsigned int K,
                                 unsigned int bm,
                                 unsigned int bn,
                                 unsigned int bk,
                                 bool use_workgroup_matrix = true,
                                 bool rearrange = true,
                                 layout_t layout_a = row_major,
                                 layout_t layout_b = col_major,
                                 layout_t layout_c = row_major,
                                 const matrix::matrix_support_info* mi = nullptr)
{
    assert(M % bm == 0);
    assert(N % bn == 0);
    assert(K % bk == 0);

    const bool use_bulk_copy = get_current_build_device().support_bulk_copy();

    unsigned int sparsity = (mi ? mi->sparse_a.sparsity() : 1u);

    if (use_workgroup_matrix)
    {
        assert((K / bk) % 4 == 0 || !use_bulk_copy); // Required for the mbarrier synchronization mechanism.
        assert(rearrange);

        mbarriers mbar(2, 2 + (sparsity != 1));

        gpu_for_group(0, (M / bm) * (N / bn), [&](gpu_uint block) {
            gpu_uint block_m = block / (N / bn);
            gpu_uint block_n = block % (N / bn);

            gpu_uint moff = block_m * bm;
            gpu_uint noff = block_n * bn;

            matrix::workgroup_matrix_c<c_float_type> mc(bm, bn);
            mc.fill(static_cast<c_float_type>(0));

            matrix::workgroup_matrix_ab<a_float_type> ma(bm, bk, 2, mi);
            matrix::workgroup_matrix_ab<b_float_type> mb(bk, bn, 2);
            unsigned int sparse_params_per_workgroup = 0;

            if (mi)
            {
                sparse_matrix<a_float_type> a(bm / warpgroup_rows(local_size(), mi->Nthreads), bk, *mi);
                sparse_params_per_workgroup =
                    a.sparse_metadata().size() * mi->Nthreads * warpgroup_rows(local_size(), mi->Nthreads);
                assert(sparse_params_per_workgroup != 0);
            }

            auto load_data = [&](gpu_uint block_k) {
                gpu_uint koff = block_k * bk;
                gpu_uint block_A = block_m * (K / bk) + block_k;

                if (use_bulk_copy)
                {
                    gpu_if(local_id() == 0)
                    {
                        bulk_copy(
                            B.begin() + (layout_b == col_major ? koff * bn + noff * K : koff * N + noff * bk),
                            B.begin()
                                + ((layout_b == col_major ? koff * bn + noff * K : koff * N + noff * bk) + bk * bn),
                            mb.storage(block_k % 2),
                            mbar(block_k % 2));
                        bulk_copy(A.begin()
                                      + (layout_a == col_major ? moff * bk + koff * M : moff * K + koff * bm)
                                            / sparsity,
                                  A.begin()
                                      + (layout_a == col_major ? moff * bk + koff * M : moff * K + koff * bm + bm * bk)
                                            / sparsity,
                                  ma.storage(block_k % 2),
                                  mbar(block_k % 2));
                        if (Asp_md)
                        {
                            bulk_copy(Asp_md->begin() + block_A * sparse_params_per_workgroup,
                                      Asp_md->begin() + (block_A + 1) * sparse_params_per_workgroup,
                                      ma.sparse_md(block_k % 2),
                                      mbar(block_k % 2));
                        }
                    }

                    ma.layout = layout_a;
                    mb.layout = layout_b;
                }
                else
                {
                    if (rearrange)
                    {
                        ma.load_async(A.begin()
                                          + (layout_a == col_major ? moff * bk + koff * M : moff * K + koff * bm)
                                                / sparsity,
                                      block_k % 2,
                                      layout_a);
                        mb.load_async(B.begin() + (layout_b == col_major ? koff * bn + noff * K : koff * N + noff * bk),
                                      block_k % 2,
                                      layout_b);
                    }
                    else
                    {
                        ma.load_async(A.begin()
                                          + (layout_a == col_major ? moff + koff * M : moff * K + koff) / sparsity,
                                      block_k % 2,
                                      layout_a,
                                      layout_a == col_major ? M : K);
                        mb.load_async(B.begin() + (layout_b == col_major ? koff + noff * K : koff * N + noff),
                                      block_k % 2,
                                      layout_b,
                                      layout_b == col_major ? K : N);
                    }
                    if (Asp_md)
                    {
                        gpu_for_local(0, ma.sparse_md_size, par_unroll(4), [&](gpu_uint k) {
                            ma.sparse_md(block_k % 2)[k] = (*Asp_md)[block_A * sparse_params_per_workgroup + k];
                        });
                    }

                    async_commit();
                }
            };

            // Start loading the first tile.
            load_data(0);

            gpu_for(0, (K / bk), [&](gpu_uint block_k) {
                gpu_if(block_k != (K / bk) - 1)
                {
                    // Trigger copying of the next tile.
                    load_data(block_k + 1);
                    if (!use_bulk_copy)
                    {
                        // And wait for the previous copy operation to finish.
                        async_wait(1);
                    }
                }
                gpu_else
                {
                    if (!use_bulk_copy)
                    {
                        // This is the last tile. Need to wait for the data transfer to finish.
                        async_wait(0);
                    }
                }
                if (use_bulk_copy)
                {
                    // bulk copy uses mbarrier synchronization mechanism. Alternating between two mbarrier
                    // objects, with 2 parity states each. No need for local_barrier.
                    mbar(block_k % 2).wait(block_k / 2);
                }
                else
                {
                    local_barrier(memory::threadgroup);
                }

                // Multiplying matrix tiles, adding the result.
                if constexpr (std::is_same_v<a_float_type, precision::fp4e2m1>)
                {
                    // fp4e2m1 uses block scaling on Nvidia GPUs. Using factor 1 for simplicity, with
                    // block_width=32.
                    matrix::warp_matrix<precision::fp8ue8m0> scale_a(mc.rows / mc.brows, ma.cols / 32);
                    matrix::warp_matrix<precision::fp8ue8m0> scale_b(mb.rows / 32, mc.cols / mc.bcols);
                    scale_a.fill(1);
                    scale_b.fill(1);
                    mc.add_product(ma, mb, block_k % 2, scale_a, scale_b);
                }
                else
                {
                    mc.add_product(ma, mb, block_k % 2);
                }

                // Barrier needed to prevent overwriting data that is currently in use.
                local_barrier(memory::threadgroup);
            });

            mc.store(C.begin() + (layout_c == col_major ? moff + noff * M : moff * N + noff),
                     layout_c,
                     layout_c == col_major ? M : N);
        });
    }
    else
    {
        // Register-only matrix multiplication.

        gpu_for_group(0, (M / bm) * (N / bn), [&](gpu_uint block) {
            gpu_uint block_m = block / (N / bn);
            gpu_uint block_n = block % (N / bn);

            gpu_uint moff = block_m * bm;
            gpu_uint noff = block_n * bn;

            matrix::warp_matrix<c_float_type> mc(bm, bn);
            mc.fill(static_cast<c_float_type>(0));

            gpu_for(0, K, bk, [&](gpu_uint koff) {
                // Loading matrix tile of Matrix A.
                matrix::warp_matrix<b_float_type> mb(bk, bn);

                if (rearrange)
                {
                    // Loading matrix tile of Matrix B.
                    mb.load(B.begin() + (layout_b == col_major ? koff * bn + noff * K : koff * N + noff * bk),
                            layout_b);
                }
                else
                {
                    // Loading matrix tile of Matrix B.
                    mb.load(B.begin() + (layout_b == col_major ? koff + noff * K : koff * N + noff),
                            layout_b,
                            layout_b == col_major ? K : N);
                }

                if (mi)
                {
                    matrix::sparse_matrix<a_float_type> ma(bm, bk, *mi);
                    if (rearrange)
                    {
                        ma.load(A.begin()
                                    + (layout_a == col_major ? moff * bk + koff * M : moff * K + koff * bm) / sparsity,
                                layout_a);
                    }
                    else
                    {
                        ma.load(A.begin() + (layout_a == col_major ? moff + koff * M : moff * K + koff) / sparsity,
                                layout_a,
                                (layout_a == col_major ? M : K) / sparsity);
                    }
                    ma.load_metadata(Asp_md->begin()
                                     + (block_m * (K / bk) + koff / bk) * ma.sparse_metadata().size() * local_size());

                    // Multiplying matrix tiles, adding the result.
                    if constexpr (std::is_same_v<a_float_type, precision::fp4e2m1>)
                    {
                        // fp4e2m1 uses block scaling on Nvidia GPUs. Using factor 1 for simplicity, with
                        // block_width=32.
                        matrix::warp_matrix<precision::fp8ue8m0> scale_a(ma.rows, ma.cols / 32);
                        matrix::warp_matrix<precision::fp8ue8m0> scale_b(mb.rows / 32, mb.cols);
                        scale_a.fill(1);
                        scale_b.fill(1);
                        mc = multiply_add(ma, mb, mc, scale_a, scale_b);
                    }
                    else
                    {
                        mc += ma * mb;
                    }
                }
                else
                {
                    matrix::warp_matrix<a_float_type> ma(bm, bk);
                    if (rearrange)
                    {
                        ma.load(A.begin() + (layout_a == col_major ? moff * bk + koff * M : moff * K + koff * bm),
                                layout_a);
                    }
                    else
                    {
                        ma.load(A.begin() + (layout_a == col_major ? moff + koff * M : moff * K + koff),
                                layout_a,
                                layout_a == col_major ? M : K);
                    }

                    // Multiplying matrix tiles, adding the result.
                    if constexpr (std::is_same_v<a_float_type, precision::fp4e2m1>)
                    {
                        // fp4e2m1 uses block scaling on Nvidia GPUs. Using factor 1 for simplicity, with
                        // block_width=32.
                        matrix::warp_matrix<precision::fp8ue8m0> scale_a(ma.rows, ma.cols / 32);
                        matrix::warp_matrix<precision::fp8ue8m0> scale_b(mb.rows / 32, mb.cols);
                        scale_a.fill(1);
                        scale_b.fill(1);
                        mc = multiply_add(ma, mb, mc, scale_a, scale_b);
                    }
                    else
                    {
                        mc += ma * mb;
                    }
                }
            });

            mc.store(C.begin() + (layout_c == col_major ? moff + noff * M : moff * N + noff),
                     layout_c,
                     layout_c == col_major ? M : N);
        });
    }
}

/**
   This function creates a suitable kernel to multiply big matrices.

   \tparam is_sparse If true, matrix A is a sparse matrix.
   \param device goopax device
   \param M,N,K Matrix sizes
   \param bm,bn,bk Sizes of the matrix tiles
   \param use_workgroup_matrix If true, multiple warps will work together and share data via local memory.
   \param rearrange Assume matrices A and B are stored in a tiled layout. This significantly increases performance.
   Required if use_workgroup_matrix==true.
   \param ls workgroup size to use when using warpgroup matrix mode.
   \param layout_a,layout_b,layout_c matrix layouts. Should be row_major for matrices A and C, and col_major for matrix
   B for best performance.
   \param mi matrix mode to use. Required for sparse matrices.

   \sa matrix.cpp
 */
template<typename a_float_type, typename b_float_type, typename c_float_type, bool is_sparse>
auto create_matmul_kernel(goopax_device device,
                          unsigned int M,
                          unsigned int N,
                          unsigned int K,
                          unsigned int bm,
                          unsigned int bn,
                          unsigned int bk,
                          bool use_workgroup_matrix = true,
                          bool rearrange = true,
                          unsigned int ls = 0,
                          layout_t layout_a = row_major,
                          layout_t layout_b = col_major,
                          layout_t layout_c = row_major,
                          const matrix::matrix_support_info* mi = nullptr)
{
    if constexpr (is_sparse)
    {
        return kernel(
            device,
            [device, mi, M, N, K, bm, bn, bk, use_workgroup_matrix, layout_a, layout_b, layout_c, rearrange](
                resource<a_float_type>& A,
                resource<b_float_type>& B,
                resource<c_float_type>& C,
                resource<unsigned int>& Asp_md) {
                create_matmul_kernel_common(A,
                                            B,
                                            C,
                                            &Asp_md,
                                            M,
                                            N,
                                            K,
                                            bm,
                                            bn,
                                            bk,
                                            use_workgroup_matrix,
                                            rearrange,
                                            layout_a,
                                            layout_b,
                                            layout_c,
                                            mi);
            },
            ls,
            0);
    }
    else
    {
        return kernel(
            device,
            [device, mi, M, N, K, bm, bn, bk, use_workgroup_matrix, layout_a, layout_b, layout_c, rearrange](
                resource<a_float_type>& A, resource<b_float_type>& B, resource<c_float_type>& C) {
                create_matmul_kernel_common(A,
                                            B,
                                            C,
                                            nullptr,
                                            M,
                                            N,
                                            K,
                                            bm,
                                            bn,
                                            bk,
                                            use_workgroup_matrix,
                                            rearrange,
                                            layout_a,
                                            layout_b,
                                            layout_c,
                                            mi);
            },
            ls,
            0);
    }
}

}
