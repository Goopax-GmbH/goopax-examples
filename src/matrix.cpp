/**
   \example matrix.cpp
   matrix multiplication example program, demonstrating the
   use of tensor core hardware acceleration
 */

#include "common/matrix.hpp"
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

// Block sizes. If 0, suitable values will be set depenting on element types.
PARAMOPT<unsigned int> BM("bm", 0);
PARAMOPT<unsigned int> BN("bn", 0);
PARAMOPT<unsigned int> BK("bk", 0);

// Matrix layouts
PARAMOPT<bool> COL_MAJOR_A("col_major_a", false);
PARAMOPT<bool> COL_MAJOR_B("col_major_b", true);
PARAMOPT<bool> COL_MAJOR_C("col_major_c", false);

// Use tiled matrix storage for A and B matrices. Must be set to true if USE_WORKGROUP_MATRIX is true.
PARAMOPT<bool> REARRANGE("rearrange", true);

// Use this workgroup size. If 0, a suitable value will be chosen.
PARAMOPT<unsigned int> LOCAL_SIZE("local_size", 0);

/*
  This function fills matrix `a` with random values, in a tiled ordering if REARRANGE is true,
  and writes the same values to the `ad` matrix for later verification in a non-tiled ordering.

  To keep it simple, we are creating the kernel as a local instance here. Normally, this is not the way to do it,
  because it may be slow.
  But since we are only calling this kernel once or twice, it shouldn't matter here.
*/
template<typename a_float_type, typename c_float_type>
void fill_random(buffer<a_float_type>& a,
                 buffer<c_float_type>& ad,
                 unsigned int brows,
                 unsigned int bcols,
                 unsigned int cols,
                 matrix::matrix_support_info::sparse_t sparse_mode = matrix::matrix_support_info::sparse_t())
{
    goopax_device device = a.get_device();

    std::random_device rd;
    WELL512_data rnd(device, device.default_global_size_max(), rd());

    kernel fill_random(
        device,
        [&rnd, sparse_mode](
            resource<a_float_type>& a, resource<c_float_type>& ad, gpu_uint brows, gpu_uint bcols, gpu_uint cols) {
            WELL512_lib rndlib(rnd);

            // Parallelizing memory access such that it is compatible with sub_byte_pointer access for int4.
            unsigned int par = max(32u / static_cast<unsigned int>(bitsize<a_float_type>::value), 1u);
            if (sparse_mode.is_sparse())
            {
                par = sparse_mode.blocksize;
            }

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

                gpu_uint nonnull_bits = (1 << sparse_mode.blocksize / sparse_mode.vector_width) - 1;
                gpu_while(popcount(nonnull_bits) != sparse_mode.non_null_per_block / sparse_mode.vector_width)
                {
                    nonnull_bits = rndlib.generate()[0] & ((1 << sparse_mode.blocksize / sparse_mode.vector_width) - 1);
                }

                for (unsigned int sub = 0; sub < par; ++sub)
                {
                    gpu_float v = rndlib.gaussian_distribution();

                    gpu_if((nonnull_bits & (1 << (sub % sparse_mode.blocksize) / sparse_mode.vector_width)) == 0)
                    {
                        v = 0;
                    }
                    if constexpr (std::is_same_v<typename make_cpu<a_float_type>::type, precision::tf32>)
                    {
                        pd[sub] = v;
                        p[sub] = static_cast<typename make_gpu<a_float_type>::type>(v);
                    }
                    else
                    {
                        pd[sub] = p[sub] = static_cast<typename make_gpu<a_float_type>::type>(v);
                    }
                }
            });
        });
    fill_random(a, ad, brows, bcols, cols);
}

template<typename A_SRC, typename A_SP>
void compress_and_create_metadata(const buffer<A_SRC>& A,
                                  buffer<A_SP>& Asp,
                                  buffer<unsigned int>& md,
                                  const matrix::matrix_support_info& mi,
                                  unsigned int M,
                                  unsigned int K,
                                  unsigned int bm,
                                  unsigned int bk,
                                  bool rearrange,
                                  unsigned int ls)
{
    Asp.assign(A.get_device(), A.size() / mi.sparse_a.sparsity());

    kernel prog(A.get_device(), [&](resource<unsigned int>& mdres) {
        unsigned int brows = matrix::warpgroup_rows(ls, mi.Nthreads);

        gpu_for_group(0, M * brows / bm * K / bk, [&](gpu_uint block) {
            gpu_uint brow = block / brows / (K / bk);
            gpu_uint bcol = block / brows % (K / bk);
            gpu_uint br = block % brow;

            matrix::sparse_matrix<A_SP> mat(bm / brows, bk, mi);

            if (rearrange)
            {
                mat.load_cast_construct_from_dense(const_resource(A).begin() + block * bk * (bm / brows));
                mat.store(resource(Asp).begin() + block * bk * (bm / brows) / mi.sparse_a.sparsity(),
                          matrix::row_major);
            }
            else
            {
                assert(brows == 1);
                mat.load_cast_construct_from_dense(const_resource(A).begin() + brow * K * bm + bcol * bk, K);
                mat.store(resource(Asp).begin() + (brow * K * bm + bcol * bk) / mi.sparse_a.sparsity(),
                          matrix::row_major,
                          K / mi.sparse_a.sparsity());
            }

            mat.store_metadata(mdres.begin() + block * mat.sparse_metadata().size() * local_size());
            md.assign(A.get_device(),
                      M / bm * K / bk * mat.sparse_metadata().size() * mi.Nthreads
                          * matrix::warpgroup_rows(ls, mi.Nthreads));
        });
    });
    prog(md);
}

template<typename a_float_type, typename b_float_type, typename c_float_type, bool is_sparse>
void run(goopax_device device, bool use_workgroup_matrix)
try
{
    // Choosing suitable matrix block sizes.
    // Larger values can improve performance, but only if there are
    // enough registers available.

    unsigned int bm;
    unsigned int bn;
    unsigned int bk;
    unsigned int ls;

    if (use_workgroup_matrix)
    {
        bm = 256;
        bn = 256;
        bk = 16;
        ls = 256;
        if (bitsize<c_float_type>::value >= 32)
        {
            bm = 128;
            bn = 128;
        }
        if (bitsize<a_float_type>::value <= 8)
        {
            bk = 32;
        }
        if (std::is_same_v<a_float_type, precision::fp4e2m1>)
        {
            bk = 64;
        }
        if (std::is_same_v<a_float_type, double>)
        {
            bm = 32;
            bn = 32;
        }
    }
    else
    {
        bm = 64;
        bn = 64;
        bk = 16;
        ls = device.default_local_size();

        if (device.max_registers() < 80)
        {
            bm = 32;
            bn = 32;
        }
        if (bitsize<a_float_type>::value == 8)
        {
            bk = 64;
        }
        if (bitsize<a_float_type>::value == 4)
        {
            bk = 64;
            if (goopax_is_integral<a_float_type>::value)
            {
                bm = 64;
                bn = 32;
            }
            else
            {
                bm = 64;
                bn = 64;
            }
        }
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
            bk *= 2;
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
    if (LOCAL_SIZE)
    {
        ls = LOCAL_SIZE;
    }

    const matrix::matrix_support_info* mi = nullptr;
    if constexpr (is_sparse)
    {
        for (auto* mi2 = device.get_matrix_support_table(); mi2; mi2 = mi2->next)
        {
            if (mi2->type_enum_a == type_enum<a_float_type>::value && mi2->type_enum_b == type_enum<b_float_type>::value
                && mi2->type_enum_c == type_enum<c_float_type>::value && mi2->is_sparse()
                && (mi2->with_block_scaling() == std::is_same_v<a_float_type, precision::fp4e2m1>)
                && bm % mi2->mnk[0] == 0 && bn % mi2->mnk[1] == 0 && bk % mi2->mnk[2] == 0)
            {
                mi = mi2;
                break;
            }
        }
        if (mi == nullptr)
        {
            return;
        }
    }

    cout << "  use_workgroup_matrix=" << use_workgroup_matrix << ". bm=" << bm << ", bn=" << bn << ", bk=" << bk
         << ", ls=" << ls;
    if (is_sparse)
    {
        if (mi)
        {
            cout << ", sparse: " << mi->sparse_a;
        }
    }
    cout << endl;

    // Matrix buffers
    buffer<typename std::conditional<std::is_same_v<a_float_type, precision::tf32> && is_sparse, float, a_float_type>::
               type>
        A(device, M * K);
    buffer<b_float_type> B(device, K * N);
    buffer<c_float_type> C(device, M * N);

    // Matrix buffers for verification
    buffer<c_float_type> Ad(device, M * K);
    buffer<c_float_type> Bd(device, K * N);

    // Filling with random numbers
    if (COL_MAJOR_A)
    {
        fill_random(A, Ad, bk, bm, M, is_sparse ? mi->sparse_a : matrix::matrix_support_info::sparse_t());
    }
    else
    {
        fill_random(A, Ad, bm, bk, K, is_sparse ? mi->sparse_a : matrix::matrix_support_info::sparse_t());
    }
    if (COL_MAJOR_B)
    {
        fill_random(B, Bd, bn, bk, K);
    }
    else
    {
        fill_random(B, Bd, bk, bn, N);
    }
    C.fill(numeric_limits<c_float_type>::quiet_NaN());

    buffer<a_float_type> Asp;
    buffer<unsigned int> sparse_metadata;

    if constexpr (is_sparse)
    {
        compress_and_create_metadata(A, Asp, sparse_metadata, *mi, M, K, bm, bk, REARRANGE, ls);
    }

    // Creating the kernel

    auto multiply = goopax::matrix::create_matmul_kernel<a_float_type, b_float_type, c_float_type, is_sparse>(
        device,
        M,
        N,
        K,
        bm,
        bn,
        bk,
        use_workgroup_matrix,
        REARRANGE,
        ls,
        COL_MAJOR_A ? matrix::col_major : matrix::row_major,
        COL_MAJOR_B ? matrix::col_major : matrix::row_major,
        COL_MAJOR_C ? matrix::col_major : matrix::row_major,
        mi);

    for (unsigned int count = 0; count < 3; ++count)
    {
        device.wait_all();

        auto time_start = steady_clock::now();
        if constexpr (is_sparse)
        {
            multiply(Asp, B, C, sparse_metadata).wait();
        }
        else
        {
            multiply(A, B, C).wait();
        }
        auto time_end = steady_clock::now();

        Tdouble time = duration_cast<duration<double>>(time_end - time_start).count();
        auto OPS = Tdouble(M()) * K() * N() * 2 / time;
        cout << "    Did matrix multiplication in " << time << " seconds. Performance: " << OPS / 1E12 << " TOPS"
             << endl;
    }
    cout << "    verifying... " << flush;

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

    VectorX<double> rwant = TA * (TB * test_vector);
    VectorX<double> rhave = TC.template cast<double>() * test_vector;

    if (VERB)
    {
        cout << "A=\n" << TA << endl;
        cout << "B=\n" << TB << endl;
        cout << "C=\n" << TC << endl;
        MatrixX<double> TC_cpu = TA * TB;
        cout << "Ccpu=\n" << TC_cpu << endl;
        cout << "diff=\n" << TC - TC_cpu << endl;
    }
    cout << "err=" << (rhave - rwant).norm() / rwant.norm() << endl << endl;
}
catch (EX::goopax_exception& e)
{
    cout << "Got exception '" << e.what() << "'" << endl;
}

template<typename a_float_type, typename b_float_type, typename c_float_type>
void run_with_types(goopax_device device)
{
    cout << "\nUsing types T_A=" << type_name(type_enum<a_float_type>::value)
         << ", T_B=" << type_name(type_enum<b_float_type>::value)
         << " and T_C=" << type_name(type_enum<c_float_type>::value) << endl;
    run<a_float_type, b_float_type, c_float_type, false>(device, false);
    run<a_float_type, b_float_type, c_float_type, false>(device, true);
    if constexpr (!std::is_same_v<a_float_type, double>)
    {
        run<a_float_type, b_float_type, c_float_type, true>(device, false);
        run<a_float_type, b_float_type, c_float_type, true>(device, true);
    }
    cout << endl;
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
        run_with_types<Tfloat, Tfloat, Tfloat>(device);
        run_with_types<precision::fp8e4m3, precision::fp8e5m2, Tfloat>(device);
        // run_with_types<precision::fp8e3m2, precision::fp8e2m3, Tfloat>(device);
        // run_with_types<precision::fp8e2m1, precision::fp8e2m1, Tfloat>(device);
        run_with_types<precision::fp4e2m1, precision::fp4e2m1, Tfloat>(device);
        run_with_types<Ttf32, Ttf32, Tfloat>(device);
        run_with_types<Tint8_t, Tint8_t, Tint>(device);
        run_with_types<precision::int4, precision::int4, Tint>(device);
        run_with_types<Tdouble, Tdouble, Tdouble>(device);

        cout << endl;
    }
}
