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
PARAMOPT<bool> COL_MAJOR_B("col_major_b", false);
PARAMOPT<bool> COL_MAJOR_C("col_major_c", false);

template<typename ab_float_type, typename c_float_type>
void run_with_types(goopax_device device)
{
    cout << "\n\nUsing types T_AB=" << goopax::pretty_typename(typeid(ab_float_type))
         << " and T_C=" << goopax::pretty_typename(typeid(c_float_type)) << endl;

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

    assert(NK % bk == 0);
    assert(NL % bl == 0);
    assert(NM % bm == 0);

    unsigned int Nthreads = device.default_local_size();

    if (!device.support_warp_matrix<ab_float_type, c_float_type>(bk, bm, bl, Nthreads))
    {
        cout << "Not supported" << endl;
        return;
    }

    buffer<ab_float_type> A(device, NK * NL);
    buffer<ab_float_type> B(device, NL * NM);
    buffer<c_float_type> C(device, NK * NM);

    cout << "memory requirements [MB]: " << (A.size() * sizeof(ab_float_type) >> 16) << " + "
         << (B.size() * sizeof(ab_float_type) >> 16) << " + " << (C.size() * sizeof(c_float_type) >> 16) << " = "
         << ((A.size() * sizeof(ab_float_type) + B.size() * sizeof(ab_float_type) + C.size() * sizeof(c_float_type))
             >> 16)
         << ", device cache: ";
    if (device.cache_size() == 0)
        cout << "unknown";
    else
        cout << (device.cache_size() >> 16);
    cout << endl;

    std::random_device rd;
    WELL512_data rnd(device, device.default_global_size_max(), rd());
    kernel fill_random(device, [&rnd](resource<ab_float_type>& a) {
        WELL512_lib rndlib(rnd);

        for_each_global(a.begin(), a.end(), [&](auto& v) {
            v = static_cast<typename make_gpu<ab_float_type>::type>(rndlib.gaussian_distribution());
        });
    });

    fill_random(A);
    fill_random(B);
    C.fill(numeric_limits<c_float_type>::quiet_NaN()).wait();

    kernel multiply(
        device,
        [bk, bl, bm, Nthreads](resource<ab_float_type>& A, resource<ab_float_type>& B, resource<c_float_type>& C) {
            gpu_for_group(0, (NK / bk) * (NM / bm), [&](gpu_uint block) {
                gpu_uint block_k;
                gpu_uint block_m;

                if (true)
                {
                    // different, possibly more cache friendly layout.
                    int ng = num_groups();
                    int sy1 = 1;
                    while (sy1 * sy1 < ng && NK % (bk * sy1 * 2) == 0 && NM % (bm * sy1 * 2) == 0)
                    {
                        sy1 *= 2;
                    }
                    cout << "ng=" << ng << ", sy1=" << sy1 << endl;

                    block_k = block / sy1 % sy1 + block / (sy1 * sy1) / (NM / bm / sy1) * sy1;
                    block_m = block % sy1 + block / (sy1 * sy1) % (NM / bm / sy1) * sy1;
                }
                else
                {
                    block_k = block / (NM / bm);
                    block_m = block % (NM / bm);
                }

                gpu_uint koff = block_k * bk;
                gpu_uint moff = block_m * bm;

                matrix::warp_matrix<c_float_type> mc(bk, bm, Nthreads);
                mc.fill(static_cast<c_float_type>(0));

                gpu_for(0, NL(), bl, [&](gpu_uint loff) {
                    matrix::warp_matrix<ab_float_type> ma(bk,
                                                          bl,
                                                          Nthreads,
                                                          A.begin()
                                                              + (COL_MAJOR_A ? koff + loff * NK() : koff * NL() + loff),
                                                          COL_MAJOR_A() ? matrix::col_major : matrix::row_major,
                                                          COL_MAJOR_A() ? NK() : NL());
                    matrix::warp_matrix<ab_float_type> mb(bl,
                                                          bm,
                                                          Nthreads,
                                                          B.begin()
                                                              + (COL_MAJOR_B ? loff + moff * NL() : loff * NM() + moff),
                                                          COL_MAJOR_B() ? matrix::col_major : matrix::row_major,
                                                          COL_MAJOR_B() ? NL() : NM());
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
        auto FLOPS = Tdouble(NK()) * NL() * NM() * 2 / time;
        cout << "Did matrix multiplication in " << time << " seconds. Performance: " << FLOPS / 1E12 << " TFLOPS"
             << endl;
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

    using ab_float_type_use =
        typename std::conditional<std::is_same_v<ab_float_type, Ttf32>, Tfloat, ab_float_type>::type;

    MatrixX<ab_float_type_use> TA;
    MatrixX<ab_float_type_use> TB;
    MatrixX<c_float_type> TC;

    {
        buffer_map MA(A);
        buffer_map MB(B);
        buffer_map MC(C);
        auto* Af = reinterpret_cast<ab_float_type_use*>(MA.data());
        auto* Bf = reinterpret_cast<ab_float_type_use*>(MB.data());

        if (COL_MAJOR_A())
        {
            TA = Map<Matrix<ab_float_type_use, Dynamic, Dynamic, ColMajor>>(Af, NK, NL);
        }
        else
        {
            TA = Map<Matrix<ab_float_type_use, Dynamic, Dynamic, RowMajor>>(Af, NK, NL);
        }
        if (COL_MAJOR_B())
        {
            TB = Map<Matrix<ab_float_type_use, Dynamic, Dynamic, ColMajor>>(Bf, NL, NM);
        }
        else
        {
            TB = Map<Matrix<ab_float_type_use, Dynamic, Dynamic, RowMajor>>(Bf, NL, NM);
        }
        if (COL_MAJOR_C())
        {
            TC = Map<Matrix<c_float_type, Dynamic, Dynamic, ColMajor>>(MC.data(), NK, NM);
        }
        else
        {
            TC = Map<Matrix<c_float_type, Dynamic, Dynamic, RowMajor>>(MC.data(), NK, NM);
        }
    }

    VectorX<double> rwant = TA.template cast<double>() * (TB.template cast<double>() * test_vector);
    VectorX<double> rhave = TC.template cast<double>() * test_vector;

    cout << "rhave.norm()=" << rhave.norm() << ", rwant.norm()=" << rwant.norm()
         << ", err=" << (rhave - rwant).norm() / rwant.norm() << endl;
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
        run_with_types<Tuint8_t, Tint>(device);
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
