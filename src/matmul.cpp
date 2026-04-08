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
PARAMOPT<unsigned int> NK("nk", 4096);
PARAMOPT<unsigned int> NL("nl", 4096);
PARAMOPT<unsigned int> NM("nm", 4096);

PARAMOPT<bool> COL_MAJOR_A("col_major_a", false);
PARAMOPT<bool> COL_MAJOR_B("col_major_b", true);
PARAMOPT<bool> COL_MAJOR_C("col_major_c", false);

template<typename ab_float_type, typename c_float_type>
void run_with_types(goopax_device device)
try
{
    cout << "\n\nUsing types T_AB=" << type_name(type_enum<ab_float_type>::value)
         << " and T_C=" << type_name(type_enum<c_float_type>::value) << endl;

    // Choosing suitable matrix block sizes.
    // Larger values can improve performance, but only if there are
    // enough registers available.
    unsigned int bk = 64;
    unsigned int bl = 16;
    unsigned int bm = 64;
    if (device.max_registers() < 80)
    {
        bk = 32;
        bm = 32;
    }
    if (get_bits<ab_float_type>::value == 4)
    {
        bl = 64;
    }

    assert(NK % bk == 0);
    assert(NL % bl == 0);
    assert(NM % bm == 0);

    unsigned int Nthreads = device.default_local_size();

    if (!device.support_warp_matrix<ab_float_type, c_float_type>(bk, bm, bl, Nthreads))
    {
        cout << "Not supported" << endl;
        return;
    }

    // Matrix buffers
    buffer<ab_float_type> A(device, NK * NL);
    buffer<ab_float_type> B(device, NL * NM);
    buffer<c_float_type> C(device, NK * NM);

    // Matrix buffers for verification
    buffer<c_float_type> Ad(device, NK * NL);
    buffer<c_float_type> Bd(device, NL * NM);

    cout << "Memory requirements [MB]: " << (A.size() * get_bits<ab_float_type>::value / 8 >> 16) << " + "
         << (B.size() * get_bits<ab_float_type>::value / 8 >> 16) << " + " << (C.size() * sizeof(c_float_type) >> 16)
         << " = "
         << ((A.size() * get_bits<ab_float_type>::value / 8 + B.size() * get_bits<ab_float_type>::value / 8
              + C.size() * sizeof(c_float_type))
             >> 16)
         << ", device cache: ";
    if (device.cache_size() == 0)
        cout << "unknown";
    else
        cout << (device.cache_size() >> 16);
    cout << "\nRegisters required: "
         << bk * bl * get_bits<ab_float_type>::value / Nthreads / 32
                + bl * bm * get_bits<ab_float_type>::value / Nthreads / 32
                + bk * bm * get_bits<c_float_type>::value / Nthreads / 32
         << " / " << device.max_registers() << endl;

    // Filling with random numbers
    std::random_device rd;
    WELL512_data rnd(device, device.default_global_size_max(), rd());
    kernel fill_random(device, [&rnd](resource<ab_float_type>& a, resource<c_float_type>& ad) {
        WELL512_lib rndlib(rnd);

        // Parallelizing memory access such that it is compatible with sub_byte_pointer access for int4.
        unsigned int par = max(32u / static_cast<unsigned int>(get_bits<ab_float_type>::value), 1u);
        gpu_for_global(0, a.size(), par, [&](gpu_uint k) {
            auto pd = ad.begin() + k;
            auto p = a.begin() + k;
            for (uint sub = 0; sub < par; ++sub)
            {
                if constexpr (std::is_same_v<ab_float_type, precision::tf32>)
                {
                    gpu_float v = rndlib.gaussian_distribution();
                    pd[sub] = v;
                    p[sub] = static_cast<typename make_gpu<ab_float_type>::type>(v);
                }
                else
                {
                    pd[sub] = p[sub] =
                        static_cast<typename make_gpu<ab_float_type>::type>(rndlib.gaussian_distribution());
                }
            }
        });
    });

    fill_random(A, Ad);
    fill_random(B, Bd);
    C.fill(numeric_limits<c_float_type>::quiet_NaN()).wait();

    // Creating the kernel
    kernel multiply(
        device,
        [bk, bl, bm, Nthreads](resource<ab_float_type>& A, resource<ab_float_type>& B, resource<c_float_type>& C) {
            gpu_for_group(0, (NK / bk) * (NM / bm), [&](gpu_uint block) {
                gpu_uint block_k = block / (NM / bm);
                gpu_uint block_m = block % (NM / bm);

                gpu_uint koff = block_k * bk;
                gpu_uint moff = block_m * bm;

                matrix::warp_matrix<c_float_type> mc(bk, bm, Nthreads);
                mc.fill(static_cast<c_float_type>(0));

                gpu_for(0, NL(), bl, [&](gpu_uint loff) {
                    // Loading matrix tile of Matrix A.
                    matrix::warp_matrix<ab_float_type> ma(bk,
                                                          bl,
                                                          Nthreads,
                                                          A.begin()
                                                              + (COL_MAJOR_A ? koff + loff * NK() : koff * NL() + loff),
                                                          COL_MAJOR_A() ? matrix::col_major : matrix::row_major,
                                                          COL_MAJOR_A() ? NK() : NL());

                    // Loading matrix tile of Matrix B.
                    matrix::warp_matrix<ab_float_type> mb(bl,
                                                          bm,
                                                          Nthreads,
                                                          B.begin()
                                                              + (COL_MAJOR_B ? loff + moff * NL() : loff * NM() + moff),
                                                          COL_MAJOR_B() ? matrix::col_major : matrix::row_major,
                                                          COL_MAJOR_B() ? NL() : NM());

                    // Multiplying matrix tiles, adding the result.
                    mc += ma * mb;
                });

                mc.store(C.begin() + (COL_MAJOR_C ? koff + moff * NK() : koff * NM() + moff),
                         COL_MAJOR_C() ? matrix::col_major : matrix::row_major,
                         COL_MAJOR_C() ? NK() : NM());
            });
        });

    for (unsigned int count = 0; count < 3; ++count)
    {
        auto time_start = steady_clock::now();
        multiply(A, B, C).wait();
        auto time_end = steady_clock::now();

        Tdouble time = duration_cast<duration<double>>(time_end - time_start).count();
        auto OPS = Tdouble(NK()) * NL() * NM() * 2 / time;
        cout << "Did matrix multiplication in " << time << " seconds. Performance: " << OPS / 1E12 << " TOPS" << endl;
    }
    cout << "verifying... " << flush;

    VectorX<double> test_vector;
    {
        std::default_random_engine generator;
        std::normal_distribution<double> distribution;
        test_vector = VectorX<double>(NM());
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

    MatrixX<double> TA = get_matrix(Ad, COL_MAJOR_A(), NK, NL);
    MatrixX<double> TB = get_matrix(Bd, COL_MAJOR_B(), NL, NM);
    MatrixX<double> TC = get_matrix(C, COL_MAJOR_C(), NK, NM);

    VectorX<double> rwant = TA * (TB * test_vector);
    VectorX<double> rhave = TC.template cast<double>() * test_vector;

    cout << "err=" << (rhave - rwant).norm() / rwant.norm() << endl;
}
catch (std::exception& e)
{
    cout << "Got exception '" << e.what() << "'" << endl;
}

int main(int argc, char** argv)
{
    init_params(argc, argv);

    for (auto device : devices(GOOPAX_DEBUG ? env_CPU : env_GPU))
    {
        cout << "running on device " << device.name() << ", env=" << device.get_envmode() << endl;
        cout << "matrix sizes: matrix<T_AB, " << NK() << ", " << NL() << "> * matrix<T_AB, " << NL() << ", " << NM()
             << "> + matrix<T_C, " << NK() << ", " << NM() << ">" << endl;

        run_with_types<Tint8_t, Tint>(device);
        run_with_types<precision::int4, Tint>(device);
        run_with_types<Thalf, Thalf>(device);
        run_with_types<Thalf, Tfloat>(device);
        run_with_types<Tbfloat16, Tfloat>(device);
        run_with_types<Tfloat, Tfloat>(device);

        if (device.support_type(Ttf32()))
        {
            run_with_types<Ttf32, Tfloat>(device);
        }
        if (device.support_type(Tdouble()))
        {
            run_with_types<Tdouble, Tdouble>(device);
        }

        cout << endl << endl;
    }
}
