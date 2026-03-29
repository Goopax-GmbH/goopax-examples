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
PARAMOPT<size_t> NK("nk", 4096);
PARAMOPT<size_t> NL("nl", 4096);
PARAMOPT<size_t> NM("nm", 4096);

PARAMOPT<bool> COL_MAJOR_A("col_major_a", false);
PARAMOPT<bool> COL_MAJOR_B("col_major_b", false);
PARAMOPT<bool> COL_MAJOR_C("col_major_c", false);

template<typename ab_float_type, typename c_float_type>
struct matmul
{
    using gpu_ab_float_type = typename make_gpu<ab_float_type>::type;
    using gpu_c_float_type = typename make_gpu<c_float_type>::type;
    goopax_device device;

    const unsigned int Nk;
    const unsigned int Nl;
    const unsigned int Nm;

    template<typename I>
    I get_index_a(I k, I l) const
    {
        if (COL_MAJOR_A())
            return k + l * Nk;
        else
            return k * Nl + l;
    }
    template<typename I>
    I get_index_b(I l, I m) const
    {
        if (COL_MAJOR_B())
            return l + m * Nl;
        else
            return l * Nm + m;
    }
    template<typename I>
    I get_index_c(I k, I m) const
    {
        if (COL_MAJOR_C())
            return k + m * Nk;
        else
            return k * Nm + m;
    }

    buffer<ab_float_type> A;
    buffer<ab_float_type> B;
    buffer<c_float_type> C;

    VectorX<double> test_vector;

    kernel<void()> kernel_simple;
    kernel<void()> kernel_tensor;

    matmul(goopax_device device0, unsigned int Nk0, unsigned int Nl0, unsigned int Nm0)
        : device(device0)
        , Nk(Nk0)
        , Nl(Nl0)
        , Nm(Nm0)
    {
        A.assign(device, Nk * Nl);
        B.assign(device, Nl * Nm);
        C.assign(device, Nk * Nm);

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

        if (device.support_type(ab_float_type()))
        {
            if constexpr (!std::is_same_v<ab_float_type, Ttf32>)
            {
                kernel_simple.assign(device, [this]() {
                    gpu_for_group(0, Nk, [&](gpu_uint k) {
                        gpu_for_local(0, Nm, [&](gpu_uint m) {
                            gpu_c_float_type sum = static_cast<c_float_type>(0);
                            gpu_for(0, Nl, [&](gpu_uint l) { sum += A[get_index_a(k, l)] * B[get_index_b(l, m)]; });
                            C[get_index_c(k, m)] = sum;
                        });
                    });
                });
            }
        }

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

        if (device.support_warp_matrix<ab_float_type, c_float_type>(bk, bm, bl))
        {
            kernel_tensor.assign(device, [this, bk, bl, bm]() {
                assert(Nk % bk == 0);
                assert(Nl % bl == 0);
                assert(Nm % bm == 0);

                gpu_for_group(0, (Nk / bk) * (Nm / bm), [&](gpu_uint block) {
                    gpu_uint block_k;
                    gpu_uint block_m;

                    if (true)
                    {
                        // different, possibly more cache friendly layout.
                        int ng = num_groups();
                        int sy1 = 1;
                        while (sy1 * sy1 < ng && Nk % (bk * sy1 * 2) == 0 && Nm % (bm * sy1 * 2) == 0)
                        {
                            sy1 *= 2;
                        }
                        cout << "ng=" << ng << ", sy1=" << sy1 << endl;

                        block_k = block / sy1 % sy1 + block / (sy1 * sy1) / (Nm / bm / sy1) * sy1;
                        block_m = block % sy1 + block / (sy1 * sy1) % (Nm / bm / sy1) * sy1;
                    }
                    else
                    {
                        block_k = block / (Nm / bm);
                        block_m = block % (Nm / bm);
                    }

                    gpu_uint koff = block_k * bk;
                    gpu_uint moff = block_m * bm;

                    matrix::warp_matrix<c_float_type> mc(bk, bm, static_cast<c_float_type>(0));

                    gpu_for(0, Nl, bl, [&](gpu_uint loff) {
                        matrix::warp_matrix<ab_float_type> ma(bk,
                                                              bl,
                                                              A.begin() + get_index_a(koff, loff),
                                                              COL_MAJOR_A() ? matrix::col_major : matrix::row_major,
                                                              COL_MAJOR_A() ? Nk : Nl);
                        matrix::warp_matrix<ab_float_type> mb(bl,
                                                              bm,
                                                              B.begin() + get_index_b(loff, moff),
                                                              COL_MAJOR_B() ? matrix::col_major : matrix::row_major,
                                                              COL_MAJOR_B() ? Nl : Nm);
                        mc = multiply_add(ma, mb, mc);
                    });

                    mc.store(C.begin() + get_index_c(koff, moff),
                             COL_MAJOR_C() ? matrix::col_major : matrix::row_major,
                             COL_MAJOR_C() ? Nk : Nm);
                });
            });
        }

        if (kernel_simple.valid() || kernel_tensor.valid())
        {
            std::random_device rd;
            WELL512_data rnd(device, device.default_global_size_max(), rd());
            kernel fill_random(device, [&rnd](resource<ab_float_type>& a) {
                WELL512_lib rndlib(rnd);

                for_each_global(a.begin(), a.end(), [&](gpu_ab_float_type& v) {
                    v = static_cast<gpu_ab_float_type>(rndlib.gaussian_distribution());
                });
            });

            fill_random(A);
            fill_random(B);

            {
                std::default_random_engine generator;
                std::normal_distribution<double> distribution;
                test_vector = VectorX<double>(Nm);
                for (double& e : test_vector)
                {
                    e = distribution(generator);
                }
            }
        }
    }

    void run(kernel<void()>& kernel_use)
    {
        C.fill(numeric_limits<c_float_type>::quiet_NaN()).wait();

        for (unsigned int count = 0; count < 3; ++count)
        {
            auto time_start = steady_clock::now();
            kernel_use().wait();
            auto time_end = steady_clock::now();

            Tdouble time = duration_cast<duration<double>>(time_end - time_start).count();
            auto FLOPS = Tdouble(NK()) * NL() * NM() * 2 / time;
            cout << "Did matrix multiplication in " << time << " seconds. Performance: " << FLOPS / 1E12 << " TFLOPS"
                 << endl;
        }
        cout << "verifying... " << flush;

        using ab_float_type_use =
            typename std::conditional<std::is_same_v<ab_float_type, Ttf32>, Tfloat, ab_float_type>::type;

        MatrixX<ab_float_type_use> TA;
        MatrixX<ab_float_type_use> TB;
        MatrixX<c_float_type> TC;

        {
            buffer_map A(this->A);
            buffer_map B(this->B);
            buffer_map C(this->C);
            auto* Af = reinterpret_cast<ab_float_type_use*>(A.data());
            auto* Bf = reinterpret_cast<ab_float_type_use*>(B.data());

            if (COL_MAJOR_A())
            {
                TA = Map<Matrix<ab_float_type_use, Dynamic, Dynamic, ColMajor>>(Af, Nk, Nl);
            }
            else
            {
                TA = Map<Matrix<ab_float_type_use, Dynamic, Dynamic, RowMajor>>(Af, Nk, Nl);
            }
            if (COL_MAJOR_B())
            {
                TB = Map<Matrix<ab_float_type_use, Dynamic, Dynamic, ColMajor>>(Bf, Nl, Nm);
            }
            else
            {
                TB = Map<Matrix<ab_float_type_use, Dynamic, Dynamic, RowMajor>>(Bf, Nl, Nm);
            }
            if (COL_MAJOR_C())
            {
                TC = Map<Matrix<c_float_type, Dynamic, Dynamic, ColMajor>>(C.data(), Nk, Nm);
            }
            else
            {
                TC = Map<Matrix<c_float_type, Dynamic, Dynamic, RowMajor>>(C.data(), Nk, Nm);
            }
        }

        VectorX<double> rwant = TA.template cast<double>() * (TB.template cast<double>() * test_vector);
        VectorX<double> rhave = TC.template cast<double>() * test_vector;

        cout << "rhave.norm()=" << rhave.norm() << ", rwant.norm()=" << rwant.norm()
             << ", err=" << (rhave - rwant).norm() / rwant.norm() << endl;
    }
};

template<typename ab_float_type, typename c_float_type>
void run_with_types(goopax_device device)
{
    cout << "\n\nUsing types T_AB=" << goopax::pretty_typename(typeid(ab_float_type))
         << " and T_C=" << goopax::pretty_typename(typeid(c_float_type)) << endl;

    matmul<ab_float_type, c_float_type> mat(device, NK(), NL(), NM());

    cout << "\nTensor kernel:" << endl;
    if (mat.kernel_tensor.get_impl() != nullptr)
    {
        mat.run(mat.kernel_tensor);
    }
    else
    {
        cout << "Not supported on this device" << endl;
    }

    cout << "\nSimple kernel:" << endl;
    if (mat.kernel_simple.get_impl() != nullptr)
    {
        mat.run(mat.kernel_simple);
    }
    else
    {
        cout << "Not supported on this device" << endl;
    }
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
