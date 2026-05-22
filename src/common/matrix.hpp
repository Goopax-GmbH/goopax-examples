#pragma once
#include <goopax>

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
            std::swap(rows_use, cols_use);
        }

        using value_type_orig = typename goopax_remove_pointer<typename make_cpu<P>::type>::type;
        using value_type_use =
            typename std::conditional<(bitsize<value_type_orig>::value < 32), int, value_type_orig>::type;

        using P_use_src =
            typename make_gpu_pointer<value_type_use, get_pointer_scope<typename make_cpu<P>::type>::value>::type;
        using P_use_dest = typename make_gpu_pointer<value_type_use, memory::threadgroup>::type;

        P_use_src ptr_use = reinterpret<P_use_src>(ptr);
        constexpr unsigned int size_factor = bitsize<value_type_use>::value / bitsize<value_type_orig>::value;
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
            std::swap(rows_use, cols_use);
        }

        using value_type_orig = typename goopax_remove_pointer<typename make_cpu<P>::type>::type;
        using value_type_use =
            typename std::conditional<(bitsize<value_type_orig>::value < 32), int, value_type_orig>::type;

        using P_use_src =
            typename make_gpu_pointer<value_type_use, get_pointer_scope<typename make_cpu<P>::type>::value>::type;
        using P_use_dest = typename make_gpu_pointer<value_type_use, memory::threadgroup>::type;

        P_use_src ptr_use = reinterpret<P_use_src>(ptr);
        constexpr unsigned int size_factor = bitsize<value_type_use>::value / bitsize<value_type_orig>::value;
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

    template<typename T_A, typename T_B>
    void add_product(const workgroup_matrix_ab<T_A>& wa, const workgroup_matrix_ab<T_B>& wb, gpu_uint slot)
    {
        gpu_uint br = warp_id_in_group() / bcols;
        gpu_uint bc = warp_id_in_group() % bcols;
        warp_matrix<T_A> a;
        if (wa.layout == row_major)
        {
            a = warp_matrix<T_A>(rows / brows, wa.cols, wa.storage(slot) + br * (rows / brows) * wa.cols, wa.layout);
        }
        else
        {
            a = warp_matrix<T_A>(rows / brows, wa.cols, wa.storage(slot) + br * (rows / brows), wa.layout, wa.rows);
        }
        warp_matrix<T_B> b;
        if (wb.layout == col_major)
        {
            b = warp_matrix<T_B>(wa.cols, cols / bcols, wb.storage(slot) + bc * (cols / bcols) * wb.rows, wb.layout);
        }
        else
        {
            b = warp_matrix<T_B>(wa.cols, cols / bcols, wb.storage(slot) + bc * (cols / bcols), wb.layout, wb.cols);
        }
        tile += a * b;
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
        warp_matrix<T_A> a;
        if (wa.layout == row_major)
        {
            a = warp_matrix<T_A>(rows / brows, wa.cols, wa.storage(slot) + br * (rows / brows) * wa.cols, wa.layout);
        }
        else
        {
            a = warp_matrix<T_A>(rows / brows, wa.cols, wa.storage(slot) + br * (rows / brows), wa.layout, wa.rows);
        }
        warp_matrix<T_B> b;
        if (wb.layout == col_major)
        {
            b = warp_matrix<T_B>(wa.cols, cols / bcols, wb.storage(slot) + bc * (cols / bcols) * wb.rows, wb.layout);
        }
        else
        {
            b = warp_matrix<T_B>(wa.cols, cols / bcols, wb.storage(slot) + bc * (cols / bcols), wb.layout, wb.cols);
        }
        tile = multiply_add(a, b, tile, block_scaling_a, block_scaling_b);
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
                       : warp_id_in_group() / bcols * tile.rows + warp_id_in_group() % bcols * (cols / bcols) * pitch),
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
    }
};

template<typename a_float_type, typename b_float_type, typename c_float_type>
kernel<void(buffer<a_float_type>& A, buffer<b_float_type>& B, buffer<c_float_type>& C)>
create_matmul_kernel(goopax_device device,
                     unsigned int M,
                     unsigned int N,
                     unsigned int K,
                     unsigned int bm,
                     unsigned int bn,
                     unsigned int bk,
                     bool use_workgroup_matrix = true,
                     bool rearrange = true,
                     unsigned int force_local_size = 0,
                     layout_t layout_a = row_major,
                     layout_t layout_b = col_major,
                     layout_t layout_c = row_major)
{
    unsigned int ls = 0;
    if (use_workgroup_matrix)
    {
        ls = 256;
    }
    if (force_local_size)
    {
        ls = force_local_size;
    }

    assert(M % bm == 0);
    assert(N % bn == 0);
    assert(K % bk == 0);

    const bool use_bulk_copy = device.support_bulk_copy();

    unsigned int Nthreads = device.default_local_size();

    if (use_workgroup_matrix)
    {
        assert((K / bk) % 4 == 0); // Required for the mbarrier synchronization mechanism.
        assert(rearrange);

        return kernel(
            device,
            [M, N, K, bm, bn, bk, use_bulk_copy, layout_a, layout_b, layout_c, rearrange](
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
                                bulk_copy(B.begin()
                                              + (layout_b == col_major ? koff * bn + noff * K : koff * N + noff * bk),
                                          B.begin()
                                              + ((layout_b == col_major ? koff * bn + noff * K : koff * N + noff * bk)
                                                 + bk * bn),
                                          mb.storage(block_k % 2),
                                          mbar(block_k % 2));
                                bulk_copy(A.begin()
                                              + (layout_a == col_major ? moff * bk + koff * M : moff * K + koff * bm),
                                          A.begin()
                                              + (layout_a == col_major ? moff * bk + koff * M
                                                                       : moff * K + koff * bm + bm * bk),
                                          ma.storage(block_k % 2),
                                          mbar(block_k % 2));
                            }

                            ma.layout = layout_a;
                            mb.layout = layout_b;
                        }
                        else
                        {
                            if (rearrange)
                            {
                                ma.load_async(
                                    A.begin() + (layout_a == col_major ? moff * bk + koff * M : moff * K + koff * bm),
                                    block_k % 2,
                                    layout_a);
                                mb.load_async(
                                    B.begin() + (layout_b == col_major ? koff * bn + noff * K : koff * N + noff * bk),
                                    block_k % 2,
                                    layout_b);
                            }
                            else
                            {
                                ma.load_async(A.begin() + (layout_a == col_major ? moff + koff * M : moff * K + koff),
                                              block_k % 2,
                                              layout_a,
                                              layout_a == col_major ? M : K);
                                mb.load_async(B.begin() + (layout_b == col_major ? koff + noff * K : koff * N + noff),
                                              block_k % 2,
                                              layout_b,
                                              layout_b == col_major ? K : N);
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
                            matrix::warp_matrix<precision::fp8ue8m0> scale_a(mc.rows/mc.brows, ma.cols / 32);
                            matrix::warp_matrix<precision::fp8ue8m0> scale_b(mb.rows / 32, mc.cols/mc.bcols);
                            scale_a.fill(static_cast<precision::fp8ue8m0>(1));
                            scale_b.fill(static_cast<precision::fp8ue8m0>(1));
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
            },
            ls,
            0);
    }
    else
    {
        // Register-only matrix multiplication.

        return kernel(
            device,
            [bm, bn, bk, M, N, K, rearrange, layout_a, layout_b, layout_c](
                resource<a_float_type>& A, resource<b_float_type>& B, resource<c_float_type>& C) {
                gpu_for_group(0, (M / bm) * (N / bn), [&](gpu_uint block) {
                    gpu_uint block_m = block / (N / bn);
                    gpu_uint block_n = block % (N / bn);

                    gpu_uint moff = block_m * bm;
                    gpu_uint noff = block_n * bn;

                    matrix::warp_matrix<c_float_type> mc(bm, bn);
                    mc.fill(static_cast<c_float_type>(0));

                    gpu_for(0, K, bk, [&](gpu_uint koff) {
                        // Loading matrix tile of Matrix A.
                        matrix::warp_matrix<a_float_type> ma(bm, bk);
                        matrix::warp_matrix<b_float_type> mb(bk, bn);

                        if (rearrange)
                        {
                            ma.load(A.begin() + (layout_a == col_major ? moff * bk + koff * M : moff * K + koff * bm),
                                    layout_a == col_major ? matrix::col_major : matrix::row_major);

                            // Loading matrix tile of Matrix B.
                            mb.load(B.begin() + (layout_b == col_major ? koff * bn + noff * K : koff * N + noff * bk),
                                    layout_b == col_major ? matrix::col_major : matrix::row_major);
                        }
                        else
                        {
                            ma.load(A.begin() + (layout_a == col_major ? moff + koff * M : moff * K + koff),
                                    layout_a,
                                    layout_a == col_major ? M : K);

                            // Loading matrix tile of Matrix B.
                            mb.load(B.begin() + (layout_b == col_major ? koff + noff * K : koff * N + noff),
                                    layout_b,
                                    layout_b == col_major ? K : N);
                        }

                        // Multiplying matrix tiles, adding the result.
                        if constexpr (std::is_same_v<a_float_type, precision::fp4e2m1>)
                        {
                            // fp4e2m1 uses block scaling on Nvidia GPUs. Using factor 1 for simplicity, with
                            // block_width=32.
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

                    mc.store(C.begin() + (layout_c == col_major ? moff + noff * M : moff * N + noff),
                             layout_c,
                             layout_c == col_major ? M : N);
                });
            });
    }
}
}
