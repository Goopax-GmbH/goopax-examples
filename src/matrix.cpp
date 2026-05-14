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

PARAMOPT<unsigned int> BM("bm", 64);
PARAMOPT<unsigned int> BN("bn", 64);
PARAMOPT<unsigned int> BK("bk", 16);

PARAMOPT<bool> COL_MAJOR_A("col_major_a", false);
PARAMOPT<bool> COL_MAJOR_B("col_major_b", true);
PARAMOPT<bool> COL_MAJOR_C("col_major_c", false);

/*
  static_assert(goopax_is_pointer<gpu_type<int*>>::value);
static_assert(goopax_is_pointer<gpu_type<goopax::cpu_pointer<signed char, memory::threadgroup>>>::value);

static_assert(!can_reinterpret_pointer_check<gpu_type<int*>, gpu_type<goopax::cpu_pointer<signed char,
memory::threadgroup>>>::value);
*/

namespace std
{
template<typename T>
gpu_ostream& operator<<(gpu_ostream& s, const vector<T>& v)
{
    s << "(";
    for (uint k = 0; k < v.size(); ++k)
    {
        if (k != 0)
            s << ", ";
        s << v[k];
    }
    s << ")";
    return s;
}
}

PARAMOPT<unsigned int> USE_LOCAL_SIZE("use_local_size", 0);
namespace goopax::matrix
{
template<typename T_A, typename T_B>
struct workgroup_matrix_product;

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

    local_mem<T> storage;

    template<typename P>
        requires pointer_valid<P>
    void load(P ptr, layout_t layout, gpu_uint pitch)
    {
        this->layout = layout;

        unsigned int rows_use = rows;
        unsigned int cols_use = cols;
        if (layout == col_major)
        {
            swap(rows_use, cols_use);
        }

        using value_type_orig = typename goopax_remove_pointer<typename make_cpu<P>::type>::type;
        using value_type_use = value_type_orig; // typename std::conditional<(get_bits<value_type_orig>::value < 32),
                                                // int, value_type_orig>::type;

        using P_use_src =
            typename make_gpu_pointer<value_type_use, get_pointer_scope<typename make_cpu<P>::type>::value>::type;
        using P_use_dest = typename make_gpu_pointer<value_type_use, memory::threadgroup>::type;

        P_use_src ptr_use = reinterpret<P_use_src>(ptr);
        constexpr unsigned int size_factor = get_bits<value_type_use>::value / get_bits<value_type_orig>::value;
        pitch /= size_factor;
        cols_use /= size_factor;

        gpu_for_local(0,
                      rows_use * cols_use,
                      par_unroll(std::min(rows * cols / local_size(),
                                          static_cast<unsigned int>(16 * 8 / get_bits<value_type_orig>::value))),
                      [&](gpu_uint k) { storage[k] = ptr_use[k + k / cols_use * (pitch - cols_use)]; });
    }

    template<typename P>
        requires pointer_valid<P>
    void load_async(P ptr, layout_t layout, gpu_uint pitch)
    {
        this->layout = layout;

        unsigned int rows_use = rows;
        unsigned int cols_use = cols;
        if (layout == col_major)
        {
            swap(rows_use, cols_use);
        }

        using value_type_orig = typename goopax_remove_pointer<typename make_cpu<P>::type>::type;
        using value_type_use = value_type_orig; // typename std::conditional<(get_bits<value_type_orig>::value < 32),
                                                // int, value_type_orig>::type;

        using P_use_src =
            typename make_gpu_pointer<value_type_use, get_pointer_scope<typename make_cpu<P>::type>::value>::type;
        using P_use_dest = typename make_gpu_pointer<value_type_use, memory::threadgroup>::type;

        P_use_src ptr_use = reinterpret<P_use_src>(ptr);
        constexpr unsigned int size_factor = get_bits<value_type_use>::value / get_bits<value_type_orig>::value;
        pitch /= size_factor;
        cols_use /= size_factor;

        gpu_for_local(0,
                      rows_use * cols_use,
                      par_unroll(std::min(rows * cols / local_size(),
                                          static_cast<unsigned int>(16 * 8 / get_bits<value_type_orig>::value))),
                      [&](gpu_uint k) {
                          async_copy(ptr_use + k + k / cols_use * (pitch - cols_use), storage.begin() + k);
                          // storage[k] = ptr_use[k + k / cols_use * (pitch - cols_use)];
                      });
    }

    template<typename P>
        requires pointer_valid<P>
    workgroup_matrix_ab(unsigned int rows, unsigned int cols, P ptr, layout_t layout, gpu_int pitch)
        : workgroup_matrix_ab(rows, cols)
    {
        load(ptr, layout, pitch);
    }

    workgroup_matrix_ab(unsigned int rows0, unsigned int cols0)
        : rows(rows0)
        , cols(cols0)
        , storage(rows * cols)
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
    template<typename T_A, typename T_B>
    workgroup_matrix_c& operator+=(const workgroup_matrix_product<T_A, T_B>& ab)
    {
        gpu_uint br = warp_id_in_group() / bcols;
        gpu_uint bc = warp_id_in_group() % bcols;
        // gpu_cout << "thread=" << global_id() << ". loading a from storage.begin() + " << br*(rows/brows)*ab.a.cols
        //<< " and b from storage.begin() + " << bc*(cols/bcols)*ab.a.rows << "\n";
        warp_matrix<T_A> a(rows / brows, ab.a.cols, ab.a.storage.begin() + br * (rows / brows) * ab.a.cols, row_major);
        warp_matrix<T_A> b(ab.a.cols, cols / bcols, ab.b.storage.begin() + bc * (cols / bcols) * ab.b.rows, col_major);
        multiply_add(a, b, tile);
        // gpu_cout << "  operator+=. before: tile=" << tile.coeffs();
        tile += a * b;
        // gpu_cout << "\n  a=" << a.coeffs()
        //<< "\n  b=" << b.coeffs()
        //<< "\n  -> tile=" << tile.coeffs() << "\n";
        return *this;
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
template<typename T_A, typename T_B>
struct workgroup_matrix_product
{
    const workgroup_matrix_ab<T_A>& a;
    const workgroup_matrix_ab<T_B>& b;
    /*
  template<typename T_C>
    operator workgroup_matrix_c<T_C>() const
    {
  workgroup_matrix<T_C> c(a.rows, b.cols);
  c.fill(static_cast<T_C>(0));
  return multiply_add(a, b, c);
    }
    */
};

/**
   operator*
   \ingroup warp_matrix
   Returns a temporary expression. The result can be used with operator+ or operator+=.
 */
template<typename T_A, typename T_B>
workgroup_matrix_product<T_A, T_B> operator*(const workgroup_matrix_ab<T_A>& a, const workgroup_matrix_ab<T_B>& b)
{
    return workgroup_matrix_product<T_A, T_B>{ a, b };
}

}

template<typename ab_float_type, typename c_float_type>
void run_with_types(goopax_device device)
try
{
    cout << "\n\nUsing types T_AB=" << type_name(type_enum<ab_float_type>::value)
         << " and T_C=" << type_name(type_enum<c_float_type>::value) << endl;

    // Choosing suitable matrix block sizes.
    // Larger values can improve performance, but only if there are
    // enough registers available.
    unsigned int bm = 64;
    unsigned int bn = 64;
    unsigned int bk = 16;
    if (device.max_registers() < 80)
    {
        bm = 32;
        bn = 32;
    }
    if (get_bits<ab_float_type>::value == 4)
    {
        bk = 64;
    }

    bm = BM();
    bn = BN();
    bk = BK();

    assert(M % bm == 0);
    assert(N % bn == 0);
    assert(K % bk == 0);

    unsigned int Nthreads = device.default_local_size();

    // Matrix buffers
    buffer<ab_float_type> A(device, M * K);
    buffer<ab_float_type> B(device, K * N);
    buffer<c_float_type> C(device, M * N);

    // Matrix buffers for verification
    buffer<c_float_type> Ad(device, M * K);
    buffer<c_float_type> Bd(device, K * N);

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
         << bm * bk * get_bits<ab_float_type>::value / Nthreads / 32
                + bk * bn * get_bits<ab_float_type>::value / Nthreads / 32
                + bm * bn * get_bits<c_float_type>::value / Nthreads / 32
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
            for (unsigned int sub = 0; sub < par; ++sub)
            {
                if constexpr (std::is_same_v<typename make_cpu<ab_float_type>::type, precision::tf32>)
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
    C.fill(numeric_limits<c_float_type>::quiet_NaN());

    // Creating the kernel
    kernel<void(buffer<ab_float_type> & A, buffer<ab_float_type> & B, buffer<c_float_type> & C)> multiply;

    if (USE_LOCAL_SIZE != 0)
    {
        const unsigned int ls = USE_LOCAL_SIZE();

        multiply.assign(
            device,
            [bm, bn, bk](resource<ab_float_type>& A, resource<ab_float_type>& B, resource<c_float_type>& C) {
                gpu_for_group(0, (M / bm) * (N / bn), [&](gpu_uint block) {
                    gpu_uint block_m = block / (N / bn);
                    gpu_uint block_n = block % (N / bn);

                    gpu_uint moff = block_m * bm;
                    gpu_uint noff = block_n * bn;

                    matrix::workgroup_matrix_c<c_float_type> mc(bm, bn);
                    mc.fill(static_cast<c_float_type>(0));

                    gpu_for(0, K(), bk, [&](gpu_uint koff) {
                        // Loading matrix tile of Matrix A.

                        local_barrier(memory::threadgroup);

                        matrix::workgroup_matrix_ab<ab_float_type> ma(bm, bk);

                        if (false)
                        {
                            ma.load_async(A.begin() + (COL_MAJOR_A ? moff + koff * M() : moff * K() + koff),
                                          COL_MAJOR_A() ? matrix::col_major : matrix::row_major,
                                          COL_MAJOR_A() ? M() : K());

                            async_commit();
                            async_wait();
                        }
                        else
                        {
                            ma.load(A.begin() + (COL_MAJOR_A ? moff + koff * M() : moff * K() + koff),
                                    COL_MAJOR_A() ? matrix::col_major : matrix::row_major,
                                    COL_MAJOR_A() ? M() : K());
                        }

                        // Loading matrix tile of Matrix B.
                        matrix::workgroup_matrix_ab<ab_float_type> mb(bk, bn);
                        mb.load(B.begin() + (COL_MAJOR_B ? koff + noff * K() : koff * N() + noff),
                                COL_MAJOR_B() ? matrix::col_major : matrix::row_major,
                                COL_MAJOR_B() ? K() : N());

                        local_barrier(memory::threadgroup);

                        // Multiplying matrix tiles, adding the result.
                        mc += ma * mb;

                        local_barrier(memory::threadgroup);
                    });

                    mc.store(C.begin() + (COL_MAJOR_C ? moff + noff * M() : moff * N() + noff),
                             COL_MAJOR_C() ? matrix::col_major : matrix::row_major,
                             COL_MAJOR_C() ? M() : N());
                });
            },
            ls,
            ls);
    }
    else
    {
        multiply.assign(
            device, [bm, bn, bk](resource<ab_float_type>& A, resource<ab_float_type>& B, resource<c_float_type>& C) {
                gpu_for_group(0, (M / bm) * (N / bn), [&](gpu_uint block) {
                    gpu_uint block_m = block / (N / bn);
                    gpu_uint block_n = block % (N / bn);

                    gpu_uint moff = block_m * bm;
                    gpu_uint noff = block_n * bn;

                    matrix::warp_matrix<c_float_type> mc(bm, bn);
                    mc.fill(static_cast<c_float_type>(0));

                    gpu_for(0, K(), bk, [&](gpu_uint koff) {
                        // Loading matrix tile of Matrix A.
                        matrix::warp_matrix<ab_float_type> ma(
                            bm,
                            bk,
                            A.begin() + (COL_MAJOR_A ? moff + koff * M() : moff * K() + koff),
                            COL_MAJOR_A() ? matrix::col_major : matrix::row_major,
                            COL_MAJOR_A() ? M() : K());

                        // Loading matrix tile of Matrix B.
                        matrix::warp_matrix<ab_float_type> mb(
                            bk,
                            bn,
                            B.begin() + (COL_MAJOR_B ? koff + noff * K() : koff * N() + noff),
                            COL_MAJOR_B() ? matrix::col_major : matrix::row_major,
                            COL_MAJOR_B() ? K() : N());

                        // Multiplying matrix tiles, adding the result.
                        mc += ma * mb;
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
        cout << "matrix sizes: matrix<T_AB, " << M() << ", " << K() << "> * matrix<T_AB, " << K() << ", " << N()
             << "> + matrix<T_C, " << M() << ", " << N() << ">" << endl;

        run_with_types<Thalf, Tfloat>(device);
        // run_with_types<Tdouble, Tdouble>(device);
        /*
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
        */

        cout << endl << endl;
    }
}
