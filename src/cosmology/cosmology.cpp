/**
   \example cosmology.cpp
   FMM N-body example program

   It is a rather complex algorithm, roughly based on
   "A short course on fast multiple methods", https://math.nyu.edu/~greengar/shortcourse_fmm.pdf,
   but with some modifications:
   - Multipoles are represented in Cartesian coordinates instead of the usual spherical harmonics.
   - A binary tree is used instead of an octree.

   The parameters are optimise for big GPUs with many registers.
   If you want to run it on smaller GPUs with <256 registers,
   you might want to reduce MULTIPOLE_ORDER to 2 or so. The precision will be worse,
   but at least it will run with usable performance.
 */

#define WITH_TIMINGS 0

#if __has_include(<opencv2/opencv.hpp>)
#include <opencv2/opencv.hpp>
#define WITH_OPENCV 1
#else
#define WITH_OPENCV 0
#endif

#include "../common/particle.hpp"
#include <SDL3/SDL_main.h>
#if WITH_TIMINGS
#include <chrono>
#endif
#include <fstream>
#include <goopax_draw/window_sdl.h>
#include <goopax_extra/output.hpp>
#include <goopax_extra/param.hpp>
#include <goopax_extra/struct_types.hpp>
#include <random>
#include <set>

using Eigen::Vector;
using Eigen::Vector3;
using Eigen::Vector4;
using namespace goopax;
using namespace std;
using chrono::duration;
using chrono::steady_clock;
using chrono::time_point;
using goopax::interface::PI;

template<typename T>
constexpr T intceil(const T a, T mod)
{
    return a + (mod - ((mod + a - 1) % mod) - 1);
}

inline uint modulo(int i, int n)
{
    return (i % n + n) % n;
}
inline gpu_uint modulo(gpu_int i, gpu_int n)
{
    return (i % n + n) % n;
}

template<typename S>
concept ostream_type = std::same_as<S, std::ostream> || std::same_as<S, goopax::gpu_ostream>;

template<ostream_type S, typename V>
#if __cpp_lib_ranges >= 201911
    requires std::ranges::input_range<V>
#else
    requires std::is_convertible<typename std::iterator_traits<typename V::iterator>::iterator_category,
                                 std::input_iterator_tag>::value
#endif
             && std::is_class<V>::value && (!is_same<V, string>::value)
             && (is_same<S, ostream>::value || is_same<S, goopax::gpu_ostream>::value)
S& operator<<(S& s, const V& v)
{
    s << "(";
    for (auto t = v.begin(); t != v.end(); ++t)
    {
        if (t != v.begin())
        {
            s << ", ";
        }
        s << *t;
    }
    s << ")";
    return s;
}

namespace std
{
template<typename A, typename B>
ostream& operator<<(ostream& s, const pair<A, B>& p)
{
    return s << "[" << p.first << "," << p.second << "]";
}
}

using signature_t = Tuint64_t;
using gpu_signature_t = typename make_gpu<signature_t>::type;

#define cout1  \
    if (false) \
    cout
#define DEBUG1 false
#define VERB1 false

Tuint log2_exact(Tsize_t a)
{
    for (unsigned int r = 0; r < 61; ++r)
    {
        if (a == (1ul << r))
        {
            return r;
        }
    }
    cout << "log2_exact: bad a=" << a << endl;
    abort();
}

#define CALC_POTENTIAL 1
#define USE_CARTESIAN_MULTIPOLES 1

#if USE_CARTESIAN_MULTIPOLES
#include "multipole_cart.hpp"
#else
#include "multipole_sh.hpp"
#endif

#include "radix_sort.hpp"
const float halflen = 4;
PARAMOPT<Tfloat> MULTIPOLE_COSTFAC("multipole_costfac", 160);
PARAMOPT<Tuint> MAX_NODESIZE("max_nodesize", 16);
PARAMOPT<Tuint> MAX_DEPTH("max_depth", 64);
PARAMOPT<Tbool> POW2_SIZEVEC("pow2_sizevec", true);

PARAMOPT<string> IC("ic", "");

PARAMOPT<Tsize_t> NUM_PARTICLES("num_particles", 1000000); // Number of particles
PARAMOPT<Tdouble> DT("dt", 5E-3);
PARAMOPT<Tdouble> MAX_DISTFAC("max_distfac", 1.2);

PARAMOPT<bool> PRECISION_TEST("precision_test", false);

constexpr unsigned int MULTIPOLE_ORDER = 4;

using GPU_DOUBLE = gpu_double;
using TDOUBLE = Tdouble;
using CTDOUBLE = Tdouble;

template<class T>
Vector<T, 4> color(T pot)
{
    T pc = log2(clamp((-pot - 0.0f) * 0.6f, 1, 15.99f));

    gpu_float slot = floor(pc);
    gpu_float x = pc - slot;
    gpu_assert(slot >= 0);
    gpu_assert(slot < 4);
    Vector<T, 4> ret =
        cond(slot == 0,
             Vector<gpu_float, 4>({ 0, x, 1 - x, 0 }),
             cond(slot == 1,
                  Vector<gpu_float, 4>({ x, 1 - x, 0, 0 }),
                  cond(slot == 2, Vector<gpu_float, 4>({ 1, x, 0, 0 }), Vector<gpu_float, 4>({ 1, 1, x, 0 }))));
    return ret;
}

template<class T>
Vector<T, 3> rot(const Vector<T, 3>& a, Tint step = 1)
{
    if (step < 0)
        return rot(a, step + 3);
    if (step == 0)
        return a;
    return rot(Vector<T, 3>({ a[1], a[2], a[0] }), step - 1);
}

template<class T>
struct treenode
{
    using goopax_struct_type = T;
    template<typename X>
    using goopax_struct_changetype = treenode<typename goopax_struct_changetype<T, X>::type>;
    using uint_type = typename change_gpu_mode<unsigned int, T>::type;
    using bool_type = typename change_gpu_mode<bool, T>::type;

    uint_type first_child;
    uint_type parent;
    uint_type pbegin;
    uint_type pend;
    // uint_type parent;
    Vector<T, 3> rcenter;
    bool_type need_split;

    template<class STREAM>
    friend STREAM& operator<<(STREAM& s, const treenode& n)
    {
        s << "[first_child=" << n.first_child << ", parent=" << n.parent << ", pbegin=" << n.pbegin
          << ", pend=" << n.pend << ", rcenter=" << n.rcenter << ", split=" << n.need_split;
        return s;
    }
};

void myassert(bool b)
{
    assert(b);
}
void myassert(gpu_bool b)
{
    gpu_assert(b);
}

template<class T>
auto calc_sig_fast(const Vector<T, 3>& x, Tuint32_t)
{
    const Tuint max_depthbits = 32;
    using sig_t = typename gettype<T>::template change<Tuint32_t>::type;

    Vector<Tuint, 3> depth = { (max_depthbits + 2) / 3, (max_depthbits + 1) / 3, (max_depthbits) / 3 };
    sig_t sig = 0;
    {
        Vector<sig_t, 3> s;
        for (Tint k = 0; k < 3; ++k)
        {
            sig_t s = sig_t(abs(x[k]) * (1.0 / (halflen * pow(2.0, Tdouble(2 - k) / 3)) * (1 << (depth[k] - 1))));
            myassert(s < (1u << depth[k]));
            s = cond(x[k] > 0, s, (1 << (depth[k] - 1)) - 1 - s);
            if (depth[k] - 1 > 8)
                s = ((s & 0x0000ff00) << 16) | (s & 0x000000ff);
            if (depth[k] - 1 > 4)
                s = ((s & 0xf0f0f0f0) << 8) | (s & 0x0f0f0f0f);
            if (depth[k] - 1 > 2)
                s = ((s & 0xcccccccc) << 4) | (s & 0x33333333);
            if (depth[k] - 1 > 1)
                s = ((s & 0xaaaaaaaa) << 2) | (s & 0x55555555);
            sig |= s << (2 - (k + (3 * 1000 - max_depthbits)) % 3);
        }
        for (Tint k = 0; k < 3; ++k)
            sig |= sig_t(x[k] > 0) << (max_depthbits - 1 - k);
    }
    return sig;
}

template<class T>
auto calc_sig_fast(const Vector<T, 3>& x, Tuint64_t)
{
    const Tuint max_depthbits = 64;
    using sig_t = typename change_gpu_mode<Tuint64_t, T>::type;

    Vector<Tuint, 3> depth = { (max_depthbits + 2) / 3, (max_depthbits + 1) / 3, (max_depthbits) / 3 };
    sig_t sig = 0;
    {
        Vector<sig_t, 3> s;
        for (Tint k = 0; k < 3; ++k)
        {
            sig_t s = sig_t(abs(x[k])
                            * static_cast<T>(1.0 / (halflen * pow(2.0, Tdouble(2 - k) / 3)) * (1 << (depth[k] - 1))));
            myassert(s < (1u << depth[k]));
            s = cond(x[k] > 0, s, (1 << (depth[k] - 1)) - 1 - s);
            if (depth[k] - 1 > 16)
                s = (gpu_uint64(s & 0xffff0000) << 32) | (s & 0x0000ffff);
            if (depth[k] - 1 > 8)
                s = ((s & 0xff00ff00ff00ff00) << 16) | (s & 0x00ff00ff00ff00ff);
            if (depth[k] - 1 > 4)
                s = ((s & 0xf0f0f0f0f0f0f0f0) << 8) | (s & 0x0f0f0f0f0f0f0f0f);
            if (depth[k] - 1 > 2)
                s = ((s & 0xcccccccccccccccc) << 4) | (s & 0x3333333333333333);
            if (depth[k] - 1 > 1)
                s = ((s & 0xaaaaaaaaaaaaaaaa) << 2) | (s & 0x5555555555555555);
            sig |= s << (2 - (k + (3 * 1000 - max_depthbits)) % 3);
        }
        for (Tint k = 0; k < 3; ++k)
            sig |= sig_t(x[k] > 0) << (max_depthbits - 1 - k);
    }
    return sig;
}

template<class signature_t, class T>
auto calc_sig(const Vector<T, 3>& x, Tuint max_depthbits)
{
    using sig_t = typename change_gpu_mode<signature_t, T>::type;
    assert(max_depthbits >= 3);
    assert(max_depthbits <= get_size<sig_t>::value * 8);
    Vector<Tuint, 3> depth = { (max_depthbits + 2) / 3, (max_depthbits + 1) / 3, (max_depthbits) / 3 };

    sig_t ret = calc_sig_fast<T>(x, signature_t()) >> (get_size<sig_t>::value * 8 - max_depthbits);

    if (DEBUG1)
    {
        sig_t cmp = 0;
        {
            Vector<sig_t, 3> s;
            for (Tint k = 0; k < 3; ++k)
            {
                cmp |= sig_t(x[k] > 0) << (max_depthbits - 1 - k);
                s[k] = sig_t(abs(x[k])
                             * static_cast<T>(1.0 / (halflen * pow(2.0, Tdouble(2 - k) / 3)) * (1 << (depth[k] - 1))));
                s[k] = cond(x[k] > 0, s[k], (1 << (depth[k] - 1)) - 1 - s[k]);
            }
            {
                Tuint k = 0;
                auto depth2 = depth;
                for (Tsize_t dest = max_depthbits - 4; dest != Tsize_t(-1); --dest)
                {
                    cmp |= (s[k] & (1 << (depth2[k] - 2))) << (dest - (depth2[k] - 2));
                    --depth2[k];
                    k = (k + 1) % 3;
                }
            }
        }

        myassert(ret == cmp);
    }
    return ret;
}

template<typename FUNC>
gpu_uint find_particle_split(const resource<pair<signature_t, Tuint>>& particles,
                             const gpu_uint begin,
                             const gpu_uint end,
                             FUNC func)
{
    gpu_uint split = begin;
    gpu_if(end != begin)
    {
        gpu_uint de = (end - begin + 2) / 2;
        gpu_while(de > 1u)
        {
            gpu_uint checksplit = min(split + de, end);
            split = cond(func(particles[checksplit - 1].first), checksplit, split);
            de = (de + 1) / 2;
        }
        gpu_uint checksplit = min(split + de, end);
        split = cond(func(particles[checksplit - 1].first), checksplit, split);
    }
    return split;
}

template<class T>
struct cosmos_base
{
    using gpu_T = typename make_gpu<T>::type;
    buffer<Vector<T, 3>> x;
    buffer<Vector<T, 3>> v;
#if CALC_POTENTIAL
    buffer<T> potential;
#endif
    buffer<T> mass;
    buffer<Vector<T, 3>> tmp;
    buffer<T> tmps; // FIXME: Reduce memory.

    buffer<pair<signature_t, Tuint>> plist1;
    buffer<pair<signature_t, Tuint>> plist2;
    const Tsize_t treesize;

    buffer<Tuint> node_memory_request;

    virtual void make_tree() = 0;

    kernel<void(buffer<Vector<T, 3>>& x,
                buffer<Vector<T, 3>>& v, // FIXME: hard link.
                Tuint size,
                T dt)>
        movefunc;

    kernel<void(const buffer<Vector<T, 3>>& x, buffer<pair<signature_t, Tuint>>& plist, Tuint size)> sort1func;

    kernel<void(const buffer<Vector<T, 3>>& in,
                buffer<Vector<T, 3>>& out,
                const buffer<pair<signature_t, Tuint>>& plist,
                Tuint size)>
        apply_vec;

    kernel<void(const buffer<T>& in, buffer<T>& out, const buffer<pair<signature_t, Tuint>>& plist, Tuint size)>
        apply_scalar;

    radix_sort<pair<signature_t, Tuint>, signature_t> Radix;

    struct vicinity_data
    {
        vector<Vector<Tint, 3>> local_vec;
        set<Tuint> local_indices;
        vector<Vector<Tint, 3>> vicinity_vec;

        static Tint make_index(Vector<Tint, 3> r)
        {
            return { r[2] * 65536 + r[1] * 256 + r[0] };
        }

        static Vector<Tint, 3> parent2child(Vector<Tint, 3> p, Tint c)
        {
            p[0] = p[0] * 2 + c;
            return rot(p);
            // return { p[1], p[2], p[0] * 2 + c };
        }
        static Vector<Tint, 3> child2parent(Vector<Tint, 3> p, Tint c)
        {
            p = rot(p, -1);
            p[0] = floor((p[0] + c) / 2.0);
            return p;
            // return { p[2], floor((p[0] + c) / 2.0), p[1] };
        }
        template<typename U>
        static Vector<U, 3> make_real(Vector<Tint, 3> v, U halflen)
        {
            return Vector<U, 3>{ static_cast<U>(v[0] * pow<2, 3>(2.0)),
                                 static_cast<U>(v[1] * pow<1, 3>(2.0)),
                                 static_cast<U>(v[2] * pow<0, 3>(2.0)) }
                   * halflen;
        }

        vicinity_data(T max_distfac)
        {
            const int max_e = ceil(max_distfac) * 2 + 1;
            assert(max_e < 128);
            Vector<Tint, 3> maxvec = { 0, 0, 0 };

            Vector<Tint, 3> ac;
            for (ac[2] = -max_e; ac[2] <= max_e; ++ac[2])
            {
                for (ac[1] = -max_e; ac[1] <= max_e; ++ac[1])
                {
                    for (ac[0] = -max_e; ac[0] <= max_e; ++ac[0])
                    {
                        Vector<Tint, 3> ac2 = 2 * ac;
                        for (auto& m : ac2)
                        {
                            m = max((int)abs(m) - 1, 0);
                        }
                        Vector<Tdouble, 3> center = make_real(ac, Tdouble(1.0));
                        Vector<Tdouble, 3> mincc = make_real(ac2, Tdouble(1.0)) / 2;

                        Tbool uc = (mincc.squaredNorm() < pow2(max_distfac * pow<-1, 3>(2.0)));

                        if (uc)
                        {

                            local_vec.push_back(ac);
                            local_indices.insert(make_index(ac));
                            if (ac.squaredNorm() != 0)
                            {
                                vicinity_vec.push_back(ac);
                            }
                            for (uint k = 0; k < 3; ++k)
                            {
                                maxvec[k] = max(maxvec[k], abs(ac[k]));
                            }
                        }
                    }
                }
            }

            for (ac[1] = -maxvec[1]; ac[1] <= maxvec[1]; ++ac[1])
            {
                for (ac[2] = -maxvec[2]; ac[2] <= maxvec[2]; ++ac[2])
                {
                    for (ac[0] = -maxvec[0]; ac[0] <= maxvec[0]; ++ac[0])
                    {
                        if (ranges::contains(vicinity_vec, ac))
                        {
                            cout << "L ";
                        }
                        else if (local_indices.contains(make_index(ac)))
                        {
                            cout << "X ";
                        }
                        else
                        {
                            cout << ". ";
                        }
                    }
                    cout << "    ";
                }
                cout << endl;
            }

            cout << "local_vec=" << local_vec << endl;
            cout << "vicinity_vec=" << vicinity_vec << endl;

            assert(max_e > maxvec[0]);
            assert(max_e > maxvec[1]);
            assert(max_e > maxvec[2]);
        }
    };

    const vicinity_data vdata;

    void step()
    {
        cout1 << "Moving." << endl;
        movefunc(x, v, x.size(), 0.5 * DT());
        cout1 << "Calculating force." << endl;
        this->make_tree();
        cout1 << "Moving." << endl;
        movefunc(x, v, x.size(), 0.5 * DT());
    }

    void make_IC(const char* filename = nullptr)
    {
        goopax_device device = x.get_device();
        size_t N = x.size();

        std::default_random_engine generator;
        std::normal_distribution<double> distribution;
        std::uniform_real_distribution<double> distribution2;

        if (filename)
        {
            v.fill({ 0, 0, 0 });
            cout << "Reading from file " << filename << endl;
#if !WITH_OPENCV
            throw std::runtime_error("Need opencv to read images");
#else
            cv::Mat image_color = cv::imread(filename);
            if (image_color.empty())
            {
                throw std::runtime_error("Failed to read image");
            }

            cv::Mat image_gray;
            cv::cvtColor(image_color, image_gray, cv::COLOR_BGR2GRAY);

            uint max_extent = max(image_gray.rows, image_gray.cols);
            Vector<double, 3> cm = { 0, 0, 0 };
            buffer_map cx(this->x);
            for (auto& r : cx)
            {
                // cout << "." << flush;
                while (true)
                {
                    for (auto& xx : r)
                    {
                        xx = distribution2(generator);
                    }
                    r[2] *= 0.1f;
                    Vector<int, 3> ri = (r * max_extent).template cast<int>();
                    if (ri[0] < image_gray.cols && ri[1] < image_gray.rows)
                    {
                        uint8_t c = image_gray.at<uint8_t>(
                            { static_cast<int>(r[0] * max_extent), static_cast<int>(r[1] * max_extent) });
                        if (distribution2(generator) * 255 < c)
                        {
                            cm += r.template cast<double>();
                            break;
                        }
                    }
                }
            }
            cm /= N;
            for (auto& r : cx)
            {
                r -= cm.cast<Tfloat>();
            }
            double extent2 = 0;
            for (auto& r : cx)
            {
                extent2 += r.squaredNorm();
            }
            extent2 /= N;
            for (auto& r : cx)
            {
                r *= 0.5 / sqrt(extent2);
                r[1] *= -1;
            }
#endif
        }
        else
        {
            Tint MODE = 2;
            if (MODE == 2)
            {
                buffer_map x(this->x);
                buffer_map v(this->v);
                for (Tuint k = 0; k < N; ++k) // Setting the initial conditions:
                { // N particles of mass 1/N each are randomly placed in a sphere of radius 1
                    Vector<T, 3> xk;
                    Vector<T, 3> vk;
                    do
                    {
                        for (Tuint i = 0; i < 3; ++i)
                        {
                            xk[i] = distribution(generator) * 0.2;
                            vk[i] = distribution(generator) * 0.2;
                        }
                    } while (xk.squaredNorm() >= 1);
                    x[k] = xk;
                    vk += Vector<T, 3>({ -xk[1], xk[0], 0 }) / (Vector<T, 3>({ -xk[1], xk[0], 0 })).norm() * 0.4f
                          * min(xk.norm() * 10, (T)1);
                    if (k < N / 2)
                        vk = -vk;
                    v[k] = vk;
                    if (k < N / 2)
                    {
                        x[k] += Vector<T, 3>{ 0.8, 0.2, 0.0 };
                        v[k] += Vector<T, 3>{ -0.4, 0.0, 0.0 };
                    }
                    else
                    {
                        x[k] -= Vector<T, 3>{ 0.8, 0.2, 0.0 };
                        v[k] += Vector<T, 3>{ 0.4, 0.0, 0.0 };
                    }
                }
            }
            else if (MODE == 3)
            {
                buffer_map x(this->x);
                for (Tsize_t p = 0; p < this->x.size(); ++p)
                {
                    for (Tint k = 0; k < 3; ++k)
                    {
                        do
                        {
                            x[p][k] = distribution(generator);
                        } while (abs(x[p][k]) >= 1);
                    }
                }
                v.fill({ 0, 0, 0 });
            }
        }
        mass.fill(1.0 / N);
    }

    void precision_test()
    {
        cout << "Doing precision test" << endl;
        goopax_device device = x.get_device();

        kernel verify(device,
                      [](const resource<Vector<T, 3>>& x,
                         const resource<T>& mass,
                         const resource<Vector<T, 3>>& force,
#if CALC_POTENTIAL
                         gather_add<TDOUBLE>& poterr,
                         const resource<T>& potential,
#endif
                         gpu_uint pnum,
                         gpu_uint np) -> gather_add<double> {
                          GPU_DOUBLE ret = 0;
#if CALC_POTENTIAL
                          poterr = 0;
#endif
                          gpu_for_global(0, np, [&](gpu_uint p) {
                              gpu_uint a = gpu_uint(gpu_uint64(pnum) * p / np);
                              Vector<GPU_DOUBLE, 3> F = { 0, 0, 0 };
                              GPU_DOUBLE P = 0;
                              gpu_for(0, pnum, [&](gpu_uint b) {
                                  auto distf = x[b] - x[a];
                                  Vector<GPU_DOUBLE, 3> dist;
                                  for (Tuint k = 0; k < 3; ++k)
                                  {
                                      dist[k] = (GPU_DOUBLE)(distf[k]);
                                  }
                                  F += static_cast<GPU_DOUBLE>(mass[b]) * dist * pow<-3, 2>(dist.squaredNorm() + 1E-20);
                                  P += cond(b == a, 0., -mass[b] * pow<-1, 2>(dist.squaredNorm() + 1E-20));
                              });
                              ret += (force[a].template cast<GPU_DOUBLE>() * (1.0 / DT()) - F).squaredNorm();
#if CALC_POTENTIAL
                              poterr += pow2(potential[a] - P);
#endif
                          });
                          return ret;
                      });

        vector<Tdouble> tottimevec;
        for (Tuint k = 0; k < 5; ++k)
        {
            v.fill({ 0, 0, 0 });

            device.wait_all();

            auto t0 = steady_clock::now();
            make_tree();
            device.wait_all();
            auto t1 = steady_clock::now();

            tottimevec.push_back(duration<double>(t1 - t0).count());
        }
        std::sort(tottimevec.begin(), tottimevec.end());
        cout << "tottime=" << tottimevec << endl;

        const Tuint np = min(x.size(), (Tuint)100);

        goopax_future<Tdouble> poterr;
        Tdouble tot = verify(x,
                             mass,
                             v,
#if CALC_POTENTIAL
                             poterr,
                             potential,
#endif
                             x.size(),
                             np)
                          .get();
        cout << "err=" << sqrt(tot / np) << ", poterr=" << sqrt(poterr.get() / np) << endl;

        ofstream PLOT("plot-" + to_string(MULTIPOLE_ORDER), std::ios::app);
        PLOT << MAX_DISTFAC() << " " << sqrt(tot / np) << " " << sqrt(poterr.get() / np) << endl;
    }

    cosmos_base(goopax_device device, Tsize_t N, Tdouble max_distfac)
        : x(device, N)
        , v(device, N)
        ,
#if CALC_POTENTIAL
        potential(device, N)
        ,
#endif
        mass(device, N)
        , tmp(device, N)
        , tmps(device, N)
        ,

        plist1(device, N)
        , plist2(device, N)
        , treesize(1.3 * N + 10000)
        , Radix(device, [](auto a, auto b) { return a.first < b.first; })
        , vdata(max_distfac)

    {
        make_IC();

        movefunc.assign(device,
                        [](resource<Vector<T, 3>>& x,
                           resource<Vector<T, 3>>& v, // FIXME: hard link.
                           gpu_uint size,
                           gpu_T dt) {
                            gpu_for_global(0, size, [&](gpu_uint k) {
                                x[k] += Vector<gpu_T, 3>(v[k]) * dt;

                                gpu_bool ok = true;
                                for (Tuint i = 0; i < 3; ++i)
                                {
                                    ok = ok && (abs(x[k][i]) <= halflen);
                                    x[k][i] = max(x[k][i], -halflen);
                                    x[k][i] = min(x[k][i], halflen);
                                }
                                gpu_if(!ok)
                                {
                                    x[k] *= (gpu_T)0.99f;
                                    v[k] = { 0, 0, 0 };
                                }
                            });
                        });

        sort1func.assign(device,
                         [](const resource<Vector<T, 3>>& x, resource<pair<signature_t, Tuint>>& plist, gpu_uint size) {
                             gpu_for_global(0, size, [&](gpu_uint k) {
                                 const auto sig = calc_sig<signature_t>(x[k], MAX_DEPTH());
                                 plist[k] = make_pair(sig, k);
                             });
                         });

        apply_vec.assign(
            device,
            [](const resource<Vector<T, 3>>& in,
               resource<Vector<T, 3>>& out,
               const resource<pair<signature_t, Tuint>>& plist,
               gpu_uint size) { gpu_for_global(0, size, [&](gpu_uint k) { out[k] = in[plist[k].second]; }); });

        apply_scalar.assign(
            device,
            [](const resource<T>& in,
               resource<T>& out,
               const resource<pair<signature_t, Tuint>>& plist,
               gpu_uint size) { gpu_for_global(0, size, [&](gpu_uint k) { out[k] = in[plist[k].second]; }); });
    }
};

template<class T, unsigned int max_multipole>
struct cosmos : public cosmos_base<T>
{
    using gpu_T = typename gettype<T>::gpu;
    using cosmos_base<T>::vdata;
    using typename cosmos_base<T>::vicinity_data;
    using cosmos_base<T>::x;
    using cosmos_base<T>::v;
    using cosmos_base<T>::mass;
    using cosmos_base<T>::potential;
    using cosmos_base<T>::plist1;
    using cosmos_base<T>::plist2;
    using cosmos_base<T>::tmp;
    using cosmos_base<T>::tmps;

    buffer<treenode<T>> tree;
    buffer<multipole<T, max_multipole>> matter_tree;
    buffer<multipole<T, max_multipole>> force_tree;
    buffer<Tuint> surrounding_buf;
    // buffer<Tuint> undone_uncles;
    unsigned int num_surrounding()
    {
        return vdata.vicinity_vec.size();
    }

    gpu_uint& surrounding_link(gpu_uint node, gpu_uint k)
    {
        gpu_assert(k < num_surrounding());
        return surrounding_buf[node * num_surrounding() + k];
    }

    // buffer<treenode<T, max_multipole>> fill3;

    kernel<void(buffer<treenode<T>>& tree, pair<Tuint, Tuint> treerange)> treecount1;

    kernel<void(buffer<treenode<T>>& tree, pair<Tuint, Tuint> treerange, buffer<Tuint>& node_memory_request)>
        treecount2;

    kernel<void(buffer<treenode<T>>& tree, pair<Tuint, Tuint> treerange)> treecount3;

    kernel<void(buffer<treenode<T>>& tree,
                const buffer<pair<signature_t, Tuint>>& particles,
                pair<Tuint, Tuint> child_treerange,
                Tuint depth,
                T halflen_sublevel)>
        treecount4;

    kernel<void(pair<Tuint, Tuint> treerange, T halflen)> make_surrounding;

    array<array<kernel<void(buffer<treenode<T>>& tree,
                            const buffer<Vector<T, 3>>& xvec,
                            const buffer<T>& massvec,
                            Tuint treebegin,
                            Tuint treeend,
                            T level_halflen)>,
                2>,
          3>
        upwards;

    kernel<void(pair<Tuint, Tuint> treerange, T level_halflen)> downwards2;

    array<
        kernel<void(
            pair<Tuint, Tuint> treerange, const buffer<Vector<T, 3>>& xvec, const buffer<T>& massvec, T level_halflen)>,
        3>
        multipole_test;

    array<kernel<void(const buffer<Vector<T, 3>>& x, buffer<Vector<T, 3>>& v, pair<Tuint, Tuint> treerange, T halflen)>,
          3>
        handle_particles;

    virtual void make_tree() final
    {
        this->sort1func(x, plist1, x.size());
        this->Radix(plist1, plist2, MAX_DEPTH());

        this->apply_vec(x, tmp, plist1, plist1.size());
        swap(x, tmp);
        this->apply_vec(v, tmp, plist1, plist1.size());
        swap(v, tmp);
        this->apply_scalar(mass, tmps, plist1, plist1.size());
        swap(mass, tmps);

        vector<pair<Tuint, Tuint>> treeranges;
        Tuint treesize = 1;
        Tuint treeoffset = 3;

#if WITH_TIMINGS
        x.get_device().wait_all();
        auto t0 = steady_clock::now();
#endif

        auto print = [&](auto& tree, pair<Tuint, Tuint> treerange) {
            const_buffer_map map(tree);
            const_buffer_map surrounding(surrounding_buf);
            for (uint k = treerange.first; k < treerange.second; ++k)
            {
                cout << k << ": " << map[k] << ". surrounding:";
                for (uint s = 0; s < num_surrounding(); ++s)
                {
                    cout << " " << surrounding[k * num_surrounding() + s];
                }
                cout << endl;
            }
        };

        {
            // tree.copy(fill3, 3, 0, 0);

            // cout << "null nodes:\n";
            // print(tree, { 0, 2 });
            // cout << "root node:\n";
            // print(tree, { 3, 4 });

            double level_halflen = halflen;
            for (Tuint depth = 0; depth < MAX_DEPTH(); ++depth)
            {
                pair<Tuint, Tuint> treerange = { treeoffset, treeoffset + treesize };

                treeranges.push_back(treerange);
                // cout << "\ndepth " << depth << endl;
                //<< ": tree: before:\n";
                //  print(tree, treerange.back());

                if (depth == MAX_DEPTH() - 1)
                    break;

                // if (depth != 0)
                {
                    make_surrounding(treerange, level_halflen);
                    // cout << "\nafter make_surrounding:\n";
                    // print(tree, treerange.back());
                }

                this->treecount1(tree, treerange).wait();

                // treecount2 and treecount3 might overshoot and handle nodes behind the end. This makes sure they
                // behave correctly.
                tree.fill(
                    { .first_child = {}, .parent = {}, .pbegin = {}, .pend = {}, .rcenter = {}, .need_split = false },
                    treerange.second,
                    treerange.first + intceil(treerange.second - treerange.first, (Tuint)treecount1.local_size()));

                surrounding_buf.fill(
                    0,
                    treerange.second * num_surrounding(),
                    (treerange.first + intceil(treerange.second - treerange.first, (Tuint)treecount1.local_size()))
                        * num_surrounding());

                this->treecount2(tree, treerange, this->node_memory_request).wait();

                // cout << "\nafter treecount1:\n";
                // print(tree, treerange.back());
                // cout << "depth=" << depth << ": node_memory_request=" << this->node_memory_request << endl;

                unsigned int num_sub2 = 0;
                {
                    buffer_map node_memory_request(this->node_memory_request);
                    Tuint next_index = treerange.second / 2;
                    for (auto& n : node_memory_request)
                    {
                        Tuint index = next_index;
                        num_sub2 += n;
                        next_index += n;
                        n = index;
                    }
                }
                // cout << "after adding: node_memory_request=" << this->node_memory_request << endl;

                if (treerange.second + intceil(2 * num_sub2, treecount1.local_size()) > tree.size())
                {
                    throw std::runtime_error("tree too small");
                }

                // cout << "\nafter treecount2:\n";
                // print(tree, treerange.back());

                treecount3(tree, treerange).wait();

                // cout << "\nafter treecount3:\n";
                // print(tree, treerange);

                treeoffset += treesize;
                treesize = num_sub2 * 2;

                treecount4(tree,
                           plist1,
                           { treeoffset, treeoffset + treesize },
                           MAX_DEPTH() - depth - 1,
                           halflen * pow(2.0, (-1 - Tint(depth)) / 3.0))
                    .wait();

                // cout << "\nafter treecount4:\n";
                // print(tree, { treeoffset, treeoffset + treesize });

                Vector<T, 3> boxsize;
                for (Tuint k = 0; k < 3; ++k)
                {
                    boxsize[k] = halflen * (pow(2.0, -Tint(depth + 2 - k) / 3 + (2.0 - k) / 3.0) + 1E-7);
                }
                cout1 << "boxsize=" << boxsize << endl;

                if (num_sub2 == 0)
                    break;

                level_halflen *= pow(2.0, -1.0 / 3);
            }
        }

#if WITH_TIMINGS
        x.get_device().wait_all();
        auto t1 = steady_clock::now();
#endif

        {
            Tdouble level_halflen = halflen * pow(2.0, -1.0 / 3 * treeranges.size());
            for (Tuint depth = treeranges.size() - 1; depth != Tuint(-1); --depth)
            {
                upwards[modulo((int)depth, 3)][depth == treeranges.size() - 1](
                    tree, x, mass, treeranges[depth].first, treeranges[depth].second, level_halflen);
                level_halflen *= pow(2.0, 1.0 / 3);

                // cout << "depth=" << depth << ": after upwards:\n";
                // print(tree, treerange[depth]);
            }
        }

#if WITH_TIMINGS
        x.get_device().wait_all();
        auto t2 = steady_clock::now();
#endif

        if (treeranges.size() > MAX_DEPTH())
        {
            cerr << "treerange.size()=" << treeranges.size() << " > MAX_DEPTH=" << MAX_DEPTH() << endl;
            throw std::runtime_error("MAX_DEPTH exceeded");
        }
        // cout << "force: null nodes:\n";
        // print(force_tree, { 0, 2 });
        // cout << "force: root node:\n";
        // print(force_tree, { 3, 4 });

        {
            double level_halflen = halflen;
            for (uint depth = 1; depth < treeranges.size(); ++depth)
            {
                level_halflen *= pow(2.0, -1.0 / 3);
                downwards2(treeranges[depth], level_halflen).wait();

                // cout << "depth=" << depth << ": after downwards:\n";
                // print(force_tree, treerange[depth]);

                handle_particles[modulo((int)depth, 3)](x, v, treeranges[depth], level_halflen).wait();

                // cout << "depth=" << depth << ": after handle_particles:\n";
                // print(force_tree, treerange[depth]);

                // cout << "\ntesting tree." << endl;
                // multipole_test[modulo((int)depth, 3)](treerange[depth], x, mass, level_halflen).wait();
            }
        }

#if WITH_TIMINGS
        x.get_device().wait_all();
        auto t3 = steady_clock::now();

        cout << "treecount: " << duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << endl;
        cout << "upwards: " << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << endl;
        cout << "downwards: " << duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " ms" << endl;
#endif
    }

    cosmos(goopax_device device, Tsize_t N, Tdouble max_distfac)
        : cosmos_base<T>(device, N, max_distfac)
        , tree(device, this->treesize)
        , matter_tree(device, this->treesize)
        , force_tree(device, this->treesize)
        , surrounding_buf(device, this->treesize * num_surrounding())
    {
        tree.fill(
            { .first_child = 0, .parent = 0, .pbegin = 0, .pend = 0, .rcenter = { 0, 0, 0 }, .need_split = false },
            0,
            4);

        {
            buffer_map tree(this->tree, 0, 4);
            tree[3].pend = plist1.size();
            tree[3].need_split = true;
        }

        matter_tree.fill(zero, 0, 4);
        force_tree.fill(zero, 0, 4);

        treecount1.assign(device, [this](resource<treenode<T>>& tree, pair<gpu_uint, gpu_uint> treerange) {
            gpu_for_global(treerange.first, treerange.second, [&](gpu_uint self) {
                gpu_uint totnum = tree[self].pend - tree[self].pbegin;
                for (uint si = 0; si < num_surrounding(); ++si)
                {
                    gpu_uint cousin = surrounding_link(self, si);
                    totnum += tree[cousin].pend - tree[cousin].pbegin;
                }
                // If true, force calculation wants to go further down the tree.
                tree[self].need_split = (totnum > MAX_NODESIZE());
            });
        });

        treecount2.assign(
            device,
            [this](
                resource<treenode<T>>& tree, pair<gpu_uint, gpu_uint> treerange, resource<Tuint>& node_memory_request) {
                gpu_uint need_child_nodes = 0;

                // Choosing loop order here in such a way that neighboring nodes in space will also be
                // neighboring nodes in memory. This loop must have the same ordering in treecount2 and
                // treecount3.
                gpu_uint group_begin = treerange.first
                                       + intceil(static_cast<gpu_uint>(gpu_uint64(treerange.second - treerange.first)
                                                                       * group_id() / num_groups()),
                                                 (gpu_uint)local_size());
                gpu_uint group_end = treerange.first
                                     + intceil(static_cast<gpu_uint>(gpu_uint64(treerange.second - treerange.first)
                                                                     * (group_id() + 1) / num_groups()),
                                               (gpu_uint)local_size());

                gpu_for(group_begin, group_end, local_size(), [&](gpu_uint group_self) {
                    gpu_uint self = group_self + local_id();
                    gpu_bool has_children = tree[self].need_split;
                    {
                        for (uint si = 0; si < num_surrounding(); ++si)
                        {
                            gpu_uint cousin = surrounding_link(self, si);
                            has_children = has_children || (tree[cousin].need_split);
                        }
                    }
                    // If has_children is true, this node must further be split. Either because force
                    // calculation wants to continue in this node, or because it is used by another node
                    // in the vicinity.
                    tree[self].first_child = (gpu_uint)has_children;
                    need_child_nodes += (gpu_uint)has_children;
                });

                need_child_nodes = work_group_reduce_add(need_child_nodes, local_size());
                gpu_if(local_id() == 0)
                {
                    // Writing memory requirements for this workgroup.
                    node_memory_request[group_id()] = need_child_nodes;
                }
            });

        this->node_memory_request.assign(device, treecount2.num_groups());

        treecount3.assign(
            device,
            [this](resource<treenode<T>>& tree, pair<gpu_uint, gpu_uint> treerange) {
                gpu_uint next_index = this->node_memory_request[group_id()];

                // Choosing loop order here in such a way that neighboring nodes in space will also be neighboring nodes
                // in memory. This loop must have the same ordering in treecount2 and treecount3.
                gpu_uint group_begin = treerange.first
                                       + intceil(static_cast<gpu_uint>(gpu_uint64(treerange.second - treerange.first)
                                                                       * group_id() / num_groups()),
                                                 (gpu_uint)local_size());
                gpu_uint group_end = treerange.first
                                     + intceil(static_cast<gpu_uint>(gpu_uint64(treerange.second - treerange.first)
                                                                     * (group_id() + 1) / num_groups()),
                                               (gpu_uint)local_size());

                gpu_for(group_begin, group_end, local_size(), [&](gpu_uint group_self) {
                    gpu_uint self = group_self + local_id();
                    gpu_bool have_child = (tree[self].first_child != 0);

                    gpu_uint index;
                    {
                        vector<gpu_uint> b = ballot(have_child, local_size());
                        gpu_uint l = local_id();
                        for (gpu_uint bt : b)
                        {
                            index = cond(l < 32u, next_index + popcount(bt & ((1u << l) - 1)), index);
                            next_index += popcount(bt);
                            l -= 32;
                        }
                    }

                    tree[self].first_child = cond(have_child, index * 2, 0u);

                    gpu_if(have_child)
                    {
                        for (Tuint childnum : { 0, 1 })
                        {
                            tree[tree[self].first_child + childnum].parent = self;
                        }
                    }
                });
            },
            this->treecount2.local_size(), // thread numbers must be the same as in treecount2.
            this->treecount2.global_size());

        treecount4.assign(device,
                          [this](resource<treenode<T>>& tree,
                                 const resource<pair<signature_t, Tuint>>& particles,
                                 pair<gpu_uint, gpu_uint> child_treerange,
                                 gpu_uint depth,
                                 gpu_T halflen_sublevel) {
                              gpu_for_global(child_treerange.first, child_treerange.second, 2, [&](gpu_uint k) {
                                  gpu_uint parent = tree[k].parent;

                                  const gpu_uint end = find_particle_split(
                                      particles, tree[parent].pbegin, tree[parent].pend, [&](gpu_signature_t id) {
                                          return ((id & (gpu_signature_t(1u) << depth)) == 0);
                                      });

                                  tree[k].pbegin = tree[parent].pbegin;
                                  tree[k].pend = end;
                                  tree[k + 1].pbegin = end;
                                  tree[k + 1].pend = tree[parent].pend;

                                  for (Tuint childnum : { 0, 1 })
                                  {
                                      gpu_uint self = k + childnum;
                                      Vector<gpu_T, 3> rcenter = tree[parent].rcenter;
                                      rcenter[0] += (childnum == 0 ? -halflen_sublevel : halflen_sublevel);
                                      tree[self].rcenter = rot(rcenter);
                                  }
                              });
                          });

        make_surrounding.assign(device, [this](pair<gpu_uint, gpu_uint> treerange, gpu_T halflen) {
            gpu_for_global(treerange.first, treerange.second, 2, [&](gpu_uint k) {
                gpu_uint parent = tree[k].parent;
                for (Tuint c = 0; c < 2; ++c)
                {
                    gpu_uint self = k + c;

                    for (Tuint si = 0; si < num_surrounding(); ++si)
                    {
                        Vector<Tint, 3> childvec = vdata.vicinity_vec[si];
                        Vector<Tint, 3> parentvec = vdata.child2parent(childvec, c);
                        auto parent_si_p = ranges::find(vdata.vicinity_vec, parentvec);

                        if (parent_si_p == vdata.vicinity_vec.end())
                        {
                            if (parentvec.squaredNorm() != 0)
                            {
                                cout << "BAD: parentvec=" << parentvec << " not 0. Perhaps max_distfac is too small."
                                     << endl;
                                throw std::runtime_error("BAD: parentvec not 0.");
                            }
                            // adding sibling
                            surrounding_link(self, si) = tree[parent].first_child + (1 - c);
                        }
                        else
                        {
                            gpu_uint uncle = surrounding_link(parent, parent_si_p - vdata.vicinity_vec.begin());
                            surrounding_link(self, si) = tree[uncle].first_child + (childvec[2] + c) % 2;
                        }
                        gpu_assert(surrounding_link(self, si) < 2u
                                   || (vdata.make_real(childvec, halflen * 2)
                                       - (tree[surrounding_link(self, si)].rcenter - tree[self].rcenter))
                                              .squaredNorm()
                                          < 1E-4f * halflen);
                    }
                }
            });
        });

        surrounding_buf.fill(0, 0, num_surrounding());
        {
            buffer_map tree(this->tree);
            tree[2].parent = 0;
            tree[3].parent = 0;
        }

        for (unsigned int mod3 = 0; mod3 < 3; ++mod3)
        {

            for (bool is_bottom : { false, true })
            {
                upwards[mod3][is_bottom].assign(
                    device,
                    [is_bottom, mod3, this](resource<treenode<T>>& tree,
                                            const resource<Vector<T, 3>>& xvec,
                                            const resource<T>& massvec,
                                            gpu_uint treebegin,
                                            gpu_uint treeend,
                                            gpu_T level_halflen) {
                        gpu_for_global(treebegin, treeend, [&](gpu_uint t) {
                            const gpu_bool is_pnode = (is_bottom || tree[t].first_child == 0);

                            multipole<gpu_T, max_multipole> Msum_r = zero;

                            gpu_for(tree[t].pbegin, cond(is_pnode, tree[t].pend, tree[t].pbegin), [&](gpu_uint p) {
                                Msum_r += multipole<gpu_T, max_multipole>::from_particle(
                                    rot(xvec[p], mod3) - tree[t].rcenter, massvec[p]);
                            });
                            gpu_for(0, cond(is_pnode, 0u, 2u), [&](gpu_uint child) {
                                const gpu_uint child_id = tree[t].first_child + child;
                                const multipole<gpu_T, max_multipole> Mcr = matter_tree[child_id];
                                Vector<gpu_T, 3> shift_r = { level_halflen * (1 - gpu_int(2 * child)), 0, 0 };
                                multipole<gpu_T, max_multipole> Mr = Mcr.rot(-1).shift_ext(shift_r);
                                Msum_r += Mr;
                            });

                            matter_tree[t] = Msum_r;
                        });
                    });
            }
            handle_particles[mod3].assign(
                device,
                [mod3, this](const resource<Vector<T, 3>>& x,
                             resource<Vector<T, 3>>& v,
                             pair<gpu_uint, gpu_uint> treerange,
                             gpu_T level_halflen) {
                    (void)level_halflen;
                    gpu_for_global(treerange.first, treerange.second, [&](gpu_uint self) {
                        gpu_if(!tree[self].need_split && tree[tree[self].parent].need_split)
                        {
                            gpu_for(tree[self].pbegin, tree[self].pend, [&](gpu_uint pa) {
                                gpu_T halflen = abs((rot(tree[tree[self].parent].rcenter) - tree[self].rcenter)[2]);
                                Vector<gpu_T, 3> halflen3 = vdata.make_real({ 1, 1, 1 }, halflen);
                                for (uint k = 0; k < 3; ++k)
                                {
                                    gpu_assert(abs((rot(x[pa], mod3) - tree[self].rcenter)[k]) < halflen3[k] * 1.001f);
                                }

                                Vector<gpu_T, 3> F = { 0, 0, 0 };
                                gpu_T P = 0;

                                for (Tuint si = 0; si < num_surrounding(); ++si)
                                {
                                    gpu_uint other_node = surrounding_link(self, si);

                                    gpu_for(tree[other_node].pbegin, tree[other_node].pend, [&](gpu_uint pb) {
                                        const Vector<gpu_T, 3> dist = x[pb] - x[pa];
                                        F += dist
                                             * (mass[pb] * pow<-1, 2>(dist.squaredNorm() + 1E-20f)
                                                * pow2(pow<-1, 2>(dist.squaredNorm() + 1E-20f)));
                                        P += -(mass[pb] * pow<-1, 2>(dist.squaredNorm() + 1E-20f));
                                    });
                                }

                                gpu_for(tree[self].pbegin, tree[self].pend, [&](gpu_uint pb) {
                                    gpu_if(pa != pb)
                                    {
                                        const Vector<gpu_T, 3> dist = x[pb] - x[pa];
                                        F += dist
                                             * (mass[pb] * pow<-1, 2>(dist.squaredNorm() + 1E-20f)
                                                * pow2(pow<-1, 2>(dist.squaredNorm() + 1E-20f)));
                                        P += -(mass[pb] * pow<-1, 2>(dist.squaredNorm() + 1E-20f));
                                    }
                                });

                                F += rot(force_tree[self].calc_force(rot(x[pa], mod3) - tree[self].rcenter),
                                         -(Tint)mod3);
#if CALC_POTENTIAL
                                P += force_tree[self].calc_loc_potential(rot(x[pa], mod3) - tree[self].rcenter);
#endif

                                v[pa] += F * (gpu_T)DT();
#if CALC_POTENTIAL
                                potential[pa] = P;
#endif

                                gpu_assert(isfinite(F.squaredNorm()));
                                gpu_assert(isfinite(P));

                                {
                                    Vector<gpu_T, 3> Ftest = zero;
                                    gpu_T Ptest = 0;

                                    multipole<gpu_T, max_multipole> Mtest = zero;

                                    gpu_for(treerange.first, treerange.second, [&](gpu_uint i) {
                                        gpu_bool have = (self == i);
                                        gpu_for(0, num_surrounding(), [&](gpu_uint si) {
                                            have = have || (i == surrounding_link(self, si));
                                        });
                                        gpu_if(!have)
                                        {
                                            Vector<gpu_T, 3> Fnode = zero;
                                            gpu_T Pnode = 0;
                                            gpu_for(tree[i].pbegin, tree[i].pend, [&](gpu_uint p) {
                                                const Vector<gpu_T, 3> dist = x[p] - x[pa];
                                                Fnode += dist
                                                         * (mass[p] * pow<-1, 2>(dist.squaredNorm() + 1E-20f)
                                                            * pow2(pow<-1, 2>(dist.squaredNorm() + 1E-20f)));
                                                Pnode += -(mass[p] * pow<-1, 2>(dist.squaredNorm() + 1E-20f));
                                            });

                                            auto M = matter_tree[i].makelocal(rot(x[pa], mod3) - tree[i].rcenter);
                                            auto Mc = matter_tree[i].makelocal(tree[self].rcenter - tree[i].rcenter);
                                            Mtest += Mc;

                                            Ftest += Fnode;
                                            Ptest += Pnode;
                                        }
                                    });

                                    Vector<gpu_T, 3> Ftest2 = zero;
                                    gpu_T Ptest2 = 0;

                                    gpu_for(tree[3].pbegin, tree[3].pend, [&](gpu_uint p) {
                                        gpu_bool have = (p >= tree[self].pbegin && p < tree[self].pend);
                                        gpu_for(0, num_surrounding(), [&](gpu_uint si) {
                                            have = have
                                                   || (p >= tree[surrounding_link(self, si)].pbegin
                                                       && p < tree[surrounding_link(self, si)].pend);
                                        });
                                        gpu_if(!have)
                                        {
                                            const Vector<gpu_T, 3> dist = x[p] - x[pa];
                                            Ftest2 += dist
                                                      * (mass[p] * pow<-1, 2>(dist.squaredNorm() + 1E-20f)
                                                         * pow2(pow<-1, 2>(dist.squaredNorm() + 1E-20f)));
                                            Ptest2 += -(mass[p] * pow<-1, 2>(dist.squaredNorm() + 1E-20f));
                                        }
                                    });
                                }
                            });
                        }
                    });
                });
        }

        downwards2.assign(device, [this](pair<gpu_uint, gpu_uint> treerange, gpu_T level_halflen) {
            gpu_assert((treerange.second - treerange.first) % 2 == 0);

            gpu_for_global(treerange.first, treerange.second, 2, [&](gpu_uint k) {
                gpu_uint parent = tree[k].parent;

                gpu_if(tree[parent].need_split)
                {
                    for (int c = 0; c < 2; ++c)
                    {
                        Vector<gpu_T, 3> shiftvec = { 0, 0, level_halflen * (2 * c - 1) };
                        gpu_uint self = k + c;
                        multipole<gpu_T, max_multipole> new_multipole = force_tree[parent].rot().shift_loc(shiftvec);

                        Tuint count = 0;
                        set<unsigned int> handled;
                        for (Tuint parent_si = 0; parent_si < num_surrounding() + 1; ++parent_si)
                        {
                            Vector<Tint, 3> unclevec;
                            if (parent_si < num_surrounding())
                            {
                                unclevec = vdata.vicinity_vec[parent_si];
                            }
                            else
                            {
                                unclevec = { 0, 0, 0 };
                            }

                            for (int c2 = 0; c2 < 2; ++c2)
                            {
                                Vector<Tint, 3> cousinvec = vdata.parent2child(unclevec, c2 - c);
                                assert(!handled.contains(vdata.make_index(cousinvec)));
                                handled.insert(vdata.make_index(cousinvec));
                                // cout << "cousinvec=" << cousinvec;
                                if (!vdata.local_indices.contains(vdata.make_index(cousinvec))
                                    && cousinvec.squaredNorm() != 0)
                                {
                                    // cout << ". updating";
                                    gpu_uint cousin =
                                        tree[(parent_si == num_surrounding() ? parent
                                                                             : surrounding_link(parent, parent_si))]
                                            .first_child
                                        + c2;
                                    new_multipole +=
                                        matter_tree[cousin].makelocal(-vdata.make_real(cousinvec, level_halflen * 2));
                                    ++count;
                                }
                                // cout << endl;
                            }
                        }
                        // cout << "downwards2: count=" << count << ", num_surrounding()=" << num_surrounding() << endl;
                        for (uint si = 0; si < num_surrounding(); ++si)
                        {
                            assert(handled.contains(vdata.make_index(vdata.vicinity_vec[si])));
                        }

                        assert(count == num_surrounding() + 1);

                        gpu_assert(isfinite(new_multipole.B[0]));
                        force_tree[self] = new_multipole;
                    }
                }
            });
        });

        for (unsigned int mod3 = 0; mod3 < 3; ++mod3)
        {
            multipole_test[mod3].assign(
                device,
                [this, mod3](pair<gpu_uint, gpu_uint> treerange,
                             const resource<Vector<T, 3>>& xvec,
                             const resource<T>& massvec,
                             gpu_T level_halflen) {
                    (void)level_halflen;
                    gpu_for_global(treerange.first, treerange.second, [&](gpu_uint k) {
                        gpu_ostream DUMP(cout);

                        gpu_uint parent = tree[k].parent;

                        DUMP << "multipole_test. tree[" << k << "]=" << tree[k];

                        {
                            multipole<gpu_T, max_multipole> new_multipole = zero;
                            gpu_for(tree[k].pbegin, tree[k].pend, [&](gpu_uint p) {
                                new_multipole += multipole<gpu_T, max_multipole>::from_particle(
                                    rot(xvec[p], mod3) - tree[k].rcenter, massvec[p]);
                            });

                            DUMP << "\nMr=" << matter_tree[k] << "ist=" << new_multipole;
                        }

                        {
                            multipole<gpu_T, max_multipole> new_multipole = zero;
                            multipole<gpu_T, max_multipole> new_multipole2 = zero;

                            gpu_for(treerange.first, treerange.second, [&](gpu_uint i) {
                                gpu_bool have = (k == i);
                                gpu_for(0, num_surrounding(), [&](gpu_uint si) {
                                    have = have || (i == surrounding_link(k, si));
                                });
                                gpu_if(!have)
                                {
                                    gpu_for(tree[i].pbegin, tree[i].pend, [&](gpu_uint p) {
                                        new_multipole +=
                                            multipole<gpu_T, max_multipole>::from_particle({ 0, 0, 0 }, mass[p])
                                                .makelocal(tree[k].rcenter - rot(xvec[p], mod3));
                                    });

                                    new_multipole2 += matter_tree[i].makelocal(tree[k].rcenter - tree[i].rcenter);
                                }
                            });
                            DUMP << "\nforce: Mr=" << force_tree[k] << ", ist=" << new_multipole
                                 << ", ist2=" << new_multipole2 << "\n";
                        }
                    });
                });
        }
    }
};

/*
  void test(Vector<Tdouble, 3> my_center, Vector<Tdouble, 3> other_center, Vector<Tdouble, 3> px, Vector<Tdouble, 3> r)
{
  constexpr uint max_multipole = 2;

  cout << "\ntest. my_center=" << my_center
       << ", other_center=" << other_center
       << ", px=" << px
       << ", r=" << r
       << endl;
  multipole<Tdouble, max_multipole> Mr = multipole<Tdouble, max_multipole>::from_particle(px - other_center, 1);
  multipole<Tdouble, max_multipole> Mf = Mr.makelocal(my_center - other_center);
  multipole<Tdouble, max_multipole> Mf2 = Mf.shift_loc(-my_center);

  const Vector<Tdouble, 3> dist = px - r;
  //cout << "dist=" << dist << endl;

  cout //<< "Mr=" << Mr << endl
       //<< "Mf=" << Mf << endl
       << "P:  ist=" << -pow<-1, 2>(dist.squaredNorm())
       << ", soll=" << Mf2.calc_loc_potential(r) << endl
       << "F: ist=" << dist * pow<-1, 2>(dist.squaredNorm()) * pow2(pow<-1, 2>(dist.squaredNorm()))
       << ", soll=" << Mf2.calc_force(r) << endl
       << endl;
}
*/

int main(int argc, char** argv)
{
    /*
      test({0,0,0}, {2,0,1}, {2.1,-0.2,0.97}, {-0.12,0.03,0.09});
      test({0.1,0,0}, {2,0,1}, {2.1,-0.2,0.97}, {-0.12,0.03,0.09});
      test({0,-0.1,0}, {2,0,1}, {2.1,-0.2,0.97}, {-0.12,0.03,0.09});
      test({0,0,0.1}, {2,0,1}, {2.1,-0.2,0.97}, {-0.12,0.03,0.09});
      test({0.21/2,-0.12/2,0.11/2}, {2,0,1}, {1.95,0.1,1.015}, {0.06,-0.015,-0.045});
      test({-0.21/2,0.12/2,-0.11/2}, {2,0,1}, {2.05,-0.1,0.985}, {-0.06,0.015,0.045});
      test({0.21,-0.12,0.11}, {2,0,1}, {1.9,0.2,1.03}, {0.12,-0.03,-0.09});
      test({-0.21,0.12,-0.11}, {2,0,1}, {2.1,-0.2,0.97}, {-0.12,0.03,0.09});
      exit(0);
    */
    init_params(argc, argv);

    unique_ptr<sdl_window> window = sdl_window::create("fmm nbody",
                                                       { 1024, 768 },
                                                       SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY,
                                                       static_cast<goopax::envmode>(env_ALL & ~env_VULKAN));
    goopax_device device = window->device;

#if GOOPAX_DEBUG
    // Increasing number of threads to be able to check for race conditions.
    device.force_global_size(192);
#endif

#if WITH_METAL
    particle_renderer Renderer(dynamic_cast<sdl_window_metal&>(*window));
    buffer<Vector3<Tfloat>> x(device, NUM_PARTICLES()); // OpenGL buffer
    buffer<Vector4<Tfloat>> color(device, NUM_PARTICLES());
#elif WITH_OPENGL
    opengl_buffer<Vector3<Tfloat>> x(device, NUM_PARTICLES()); // OpenGL buffer
    opengl_buffer<Vector4<Tfloat>> color(device, NUM_PARTICLES());
#else
    buffer<Vector3<Tfloat>> x(device, NUM_PARTICLES());
    buffer<Vector4<Tfloat>> color(device, NUM_PARTICLES());
#endif

    using T = Tfloat;

    cosmos<T, MULTIPOLE_ORDER> Cosmos(device, NUM_PARTICLES(), MAX_DISTFAC());

    if (argc >= 2)
    {
        Cosmos.make_IC(argv[1]);
    }

    if (PRECISION_TEST())
    {
        Cosmos.precision_test();
        return 0;
    }

    kernel set_colors(device, [&](const resource<Vector<T, 3>>& cx) {
        gpu_for_global(0, x.size(), [&](gpu_uint k) {
            color[k] = ::color(static_cast<gpu_float>(Cosmos.potential[k]));
            x[k] = cx[k].cast<gpu_float>();
            // Tweaking z coordinate to use potential for depth testing.
            // Particles are displayed according to their x and y coordinates.
            // If multiple particles are drawn at the same pixel, the one with the
            // highest potential will be shown.
            x[k][2] = static_cast<gpu_float>(-Cosmos.potential[k]) * 0.01f;
        });
    });

    bool quit = false;
    while (!quit)
    {
        while (auto e = window->get_event())
        {
            if (e->type == SDL_EVENT_QUIT)
            {
                quit = true;
            }
            else if (e->type == SDL_EVENT_KEY_DOWN)
            {
                switch (e->key.key)
                {
                    case SDLK_ESCAPE:
                        quit = true;
                        break;
                    case SDLK_F:
                        window->toggle_fullscreen();
                        break;
                };
            }
        }

        static auto frametime = steady_clock::now();
        static Tint framecount = 0;

        Cosmos.step();

        auto now = steady_clock::now();
        ++framecount;
        if (now - frametime > chrono::seconds(1))
        {
            stringstream title;
            Tdouble rate = framecount / chrono::duration<double>(now - frametime).count();
            title << "N-body. N=" << x.size() << ", " << rate << " fps, device=" << device.name();
            string s = title.str();
            SDL_SetWindowTitle(window->window, s.c_str());
            framecount = 0;
            frametime = now;
        }

        set_colors(Cosmos.x);

#if WITH_METAL
        Renderer.render(x);
#elif WITH_OPENGL
        render(window->window, x, &color);
        SDL_GL_SwapWindow(window->window);
#else
        cout << "x=" << x << endl;
#endif
    }
    return 0;
}
