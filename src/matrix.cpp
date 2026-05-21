/**
   \example matmul.cpp
   matrix multiplication example program, demonstrating the
   use of tensor core hardware acceleration
 */

#include "common/matrix.hpp"
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

// Block sizes. If 0, suitable values will be set depenting on element types.
PARAMOPT<unsigned int> BM("bm", 0);
PARAMOPT<unsigned int> BN("bn", 0);
PARAMOPT<unsigned int> BK("bk", 0);

// Matrix layouts
PARAMOPT<bool> COL_MAJOR_A("col_major_a", false);
PARAMOPT<bool> COL_MAJOR_B("col_major_b", true);
PARAMOPT<bool> COL_MAJOR_C("col_major_c", false);

// If true, matrix multiplication will be done on workgroup level, reducing memory bandwidth.
PARAMOPT<bool> USE_WORKGROUP_MATRIX("use_workgroup_matrix", true);

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
    cout << "\nUsing types T_A=" << type_name(type_enum<a_float_type>::value)
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
        bm = 256;
        bn = 256;
        bk = 16;
        if (get_bits<c_float_type>::value >= 32)
        {
            bm = 128;
            bn = 128;
        }
        if (get_bits<a_float_type>::value <= 8)
        {
            bk = 32;
        }
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

    cout << "  bm=" << bm << ", bn=" << bn << ", bk=" << bk << endl;

    // Matrix buffers
    buffer<a_float_type> A(device, M * K);
    buffer<b_float_type> B(device, K * N);
    buffer<c_float_type> C(device, M * N);

    // Matrix buffers for verification
    buffer<c_float_type> Ad(device, M * K);
    buffer<c_float_type> Bd(device, K * N);

    // Filling with random numbers
    std::random_device rd;
    WELL512_data rnd(device, device.default_global_size_max(), rd());

    fill_random(rnd, A, Ad, bm, bk, K);
    fill_random(rnd, B, Bd, bn, bk, K);
    C.fill(numeric_limits<c_float_type>::quiet_NaN());

    // Creating the kernel
    auto multiply = create_matmul_kernel<a_float_type, b_float_type, c_float_type>(
        device,
        M,
        N,
        K,
        bm,
        bn,
        bk,
        USE_WORKGROUP_MATRIX,
        REARRANGE,
        LOCAL_SIZE,
        COL_MAJOR_A ? matrix::col_major : matrix::row_major,
        COL_MAJOR_B ? matrix::col_major : matrix::row_major,
        COL_MAJOR_C ? matrix::col_major : matrix::row_major);

    for (unsigned int count = 0; count < 3; ++count)
    {
        device.wait_all();

        auto time_start = steady_clock::now();
        multiply(A, B, C).wait();
        auto time_end = steady_clock::now();

        Tdouble time = duration_cast<duration<double>>(time_end - time_start).count();
        auto OPS = Tdouble(M()) * K() * N() * 2 / time;
        cout << "  Did matrix multiplication in " << time << " seconds. Performance: " << OPS / 1E12 << " TOPS" << endl;
    }
    cout << "  verifying... " << flush;

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

    cout << "err=" << (rhave - rwant).norm() / rwant.norm() << endl;
}
catch (EX::goopax_exception& e)
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

        cout << endl;
    }
}
