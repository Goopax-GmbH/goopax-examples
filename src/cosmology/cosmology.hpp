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
PARAMOPT<Tuint> MAX_DEPTH("max_depth", 64);

constexpr unsigned int MULTIPOLE_ORDER = 4;

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
    Vector<T, 3> rcenter;
    bool_type need_split;
    bool_type is_partnode;

    template<class STREAM>
    friend STREAM& operator<<(STREAM& s, const treenode& n)
    {
        s << "[first_child=" << n.first_child << ", parent=" << n.parent << ", pbegin=" << n.pbegin
          << ", pend=" << n.pend << ", rcenter=" << n.rcenter << ", split=" << n.need_split
          << ", is_partnode=" << n.is_partnode;
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

struct vicinity_data
{
    vector<Vector<Tint, 3>> local_vec;
    set<Tuint> local_indices;
    vector<Vector<Tint, 3>> vicinity_vec;
    const double max_distfac;

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

    vicinity_data(double max_distfac0)
        : max_distfac(max_distfac0)
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
    buffer<Vector<T, 3>> force;
    buffer<T> tmps; // FIXME: Reduce memory.

    buffer<pair<signature_t, Tuint>> plist1;
    buffer<pair<signature_t, Tuint>> plist2;
    const vicinity_data vdata;

    const size_t max_treesize;
    const unsigned int max_nodesize;
    Tuint num_particle_calls;

    buffer<treenode<T>> tree;

    buffer<Tuint> split_indices;
    buffer<Tuint> partnode_indices;
    buffer<Tuint> surrounding_buf;

    vector<pair<Tuint, Tuint>> treeranges;
    vector<pair<Tuint, Tuint>> split_index_ranges;
    vector<pair<Tuint, Tuint>> partnode_index_ranges;

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

    buffer<array<Tuint, 3>> node_memory_request;

    virtual void compute_force() = 0;

    kernel<void(pair<Tuint, Tuint> treerange)> treecount1;

    kernel<void(pair<Tuint, Tuint> treerange, buffer<array<Tuint, 3>>& node_memory_request)> treecount2;

    kernel<void(pair<Tuint, Tuint> treerange)> treecount3;

    kernel<void(const buffer<pair<signature_t, Tuint>>& particles,
                pair<Tuint, Tuint> child_treerange,
                Tuint depth,
                T halflen_sublevel)>
        treecount4;

    kernel<void(pair<Tuint, Tuint> treerange, T halflen)> make_surrounding;

    kernel<void(T dt)> movefunc;

    kernel<void(T dt)> kick;

    kernel<void(buffer<pair<signature_t, Tuint>>& plist, Tuint size)> sort1func;

    kernel<void(const buffer<Vector<T, 3>>& in,
                buffer<Vector<T, 3>>& out,
                const buffer<pair<signature_t, Tuint>>& plist,
                Tuint size)>
        apply_vec;

    kernel<void(const buffer<T>& in, buffer<T>& out, const buffer<pair<signature_t, Tuint>>& plist, Tuint size)>
        apply_scalar;

    kernel<double(
#if CALC_POTENTIAL
        goopax_future<double>& poterr,
#endif
        Tuint pnum)>
        verify;

    radix_sort<pair<signature_t, Tuint>, signature_t> Radix;

    virtual void make_tree() final
    {
        this->sort1func(plist1, x.size());
        this->Radix(plist1, plist2, MAX_DEPTH());

        assert(x.size() == force.size());
        assert(x.size() == plist1.size());
        this->apply_vec(x, force, plist1, plist1.size());
        x.copy(force);
        this->apply_vec(v, force, plist1, plist1.size());
        v.copy(force);
        this->apply_scalar(mass, tmps, plist1, plist1.size());
        mass.copy(tmps);

        Tuint treesize = 1;
        Tuint treeoffset = 3;
        Tuint split_index_offset = 0;
        Tuint partnode_index_offset = 0;

        this->treeranges.clear();
        this->split_index_ranges.clear();
        this->partnode_index_ranges.clear();

#if WITH_TIMINGS
        x.get_device().wait_all();
        auto t0 = steady_clock::now();
#endif

        {
            double level_halflen = halflen;
            for (Tuint depth = 0; depth < MAX_DEPTH(); ++depth)
            {
                pair<Tuint, Tuint> treerange = { treeoffset, treeoffset + treesize };

                this->treeranges.push_back(treerange);

                if (depth == MAX_DEPTH() - 1)
                    break;

                make_surrounding(treerange, level_halflen);

                if (depth != 0)
                {
                    // cout << "calling treecount1. treerange=" << treerange << endl;
                    this->treecount1(treerange).wait();
                }

                // The treecount2 and treecount3 loops may overshoot behind the end. This makes sure that they behave
                // nicely.
                tree.fill({ .first_child = {},
                            .parent = {},
                            .pbegin = {},
                            .pend = {},
                            .rcenter = {},
                            .need_split = false,
                            .is_partnode = false },
                          treerange.second,
                          treerange.first
                              + intceil(treerange.second - treerange.first, (Tuint)treecount1.local_size()));

                this->surrounding_buf.fill(
                    0,
                    treerange.second * num_surrounding(),
                    (treerange.first + intceil(treerange.second - treerange.first, (Tuint)treecount1.local_size()))
                        * num_surrounding());

                this->treecount2(treerange, this->node_memory_request).wait();

                unsigned int num_sub2 = 0;
                {
                    Tuint next_split_index = split_index_offset;
                    Tuint next_partnode_index = partnode_index_offset;
                    buffer_map node_memory_request(this->node_memory_request);
                    Tuint next_index = treerange.second / 2;
                    for (auto& n : node_memory_request)
                    {
                        Tuint index = next_index;
                        num_sub2 += n[0];
                        next_index += n[0];
                        n[0] = index;

                        Tuint split_index = next_split_index;
                        next_split_index += n[1];
                        n[1] = split_index;

                        Tuint partnode_index = next_partnode_index;
                        next_partnode_index += n[2];
                        n[2] = partnode_index;
                    }
                    split_index_ranges.push_back({ split_index_offset, next_split_index });
                    split_index_offset = next_split_index;

                    partnode_index_ranges.push_back({ partnode_index_offset, next_partnode_index });
                    partnode_index_offset = next_partnode_index;
                }

                if (treerange.second + intceil(2 * num_sub2, treecount1.local_size()) > tree.size())
                {
                    throw std::runtime_error("tree too small");
                }

                treecount3(treerange).wait();

                treeoffset += treesize;
                treesize = num_sub2 * 2;

                treecount4(plist1,
                           { treeoffset, treeoffset + treesize },
                           MAX_DEPTH() - depth - 1,
                           halflen * pow(2.0, (-1 - Tint(depth)) / 3.0))
                    .wait();

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
            // cout << "tree size: " << treeoffset << " / " << tree.size() << "\nsplit_index size: " <<
            // split_index_offset
            //<< "\npartnode size: " << partnode_index_offset << endl;
        }
        if (treeranges.size() > MAX_DEPTH())
        {
            cerr << "treerange.size()=" << treeranges.size() << " > MAX_DEPTH=" << MAX_DEPTH() << endl;
            throw std::runtime_error("MAX_DEPTH exceeded");
        }

#if WITH_TIMINGS
        x.get_device().wait_all();
        auto t1 = steady_clock::now();
        cout << "treecount: " << duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << endl;
#endif
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
        // cout << "Doing precision test" << endl;
        const Tuint np = min(x.size(), (Tuint)100);

        goopax_future<double> poterr;
        Tdouble tot = verify(
#if CALC_POTENTIAL
                          poterr,
#endif
                          np)
                          .get();
        cout << "err=" << sqrt(tot / np) << ", poterr=" << sqrt(poterr.get() / np) << endl;

        static ofstream PLOT("plot");

        PLOT << sqrt(tot / np) << " " << sqrt(poterr.get() / np) << " " << endl;
    }

    cosmos_base(goopax_device device, Tsize_t N, size_t max_treesize0, Tdouble max_distfac, unsigned int max_nodesize0)
        : x(device, N)
        , v(device, N)
        ,
#if CALC_POTENTIAL
        potential(device, N)
        ,
#endif
        mass(device, N)
        , force(device, N)
        , tmps(device, N)
        , plist1(device, N)
        , plist2(device, N)
        , vdata(max_distfac)
        , max_treesize(max_treesize0)
        , max_nodesize(max_nodesize0)
        , tree(device, max_treesize)
        , split_indices(device, max_treesize)
        , partnode_indices(device, max_treesize)
        , surrounding_buf(device, max_treesize * num_surrounding())
        , Radix(device, [](auto a, auto b) { return a.first < b.first; })

    {
        tree.fill({ .first_child = 0,
                    .parent = 0,
                    .pbegin = 0,
                    .pend = 0,
                    .rcenter = { 0, 0, 0 },
                    .need_split = false,
                    .is_partnode = false },
                  0,
                  4);

        {
            buffer_map tree(this->tree, 0, 4);
            tree[3].pend = plist1.size();
            tree[3].need_split = true;
        }

        treecount1.assign(device, [this](pair<gpu_uint, gpu_uint> treerange) {
            gpu_for_global(treerange.first, treerange.second, [&](gpu_uint self) {
                gpu_uint totnum = tree[self].pend - tree[self].pbegin;
                for (uint si = 0; si < num_surrounding(); ++si)
                {
                    gpu_uint cousin = surrounding_link(self, si);
                    totnum += tree[cousin].pend - tree[cousin].pbegin;
                }
                // If true, force calculation wants to go further down the tree.
                tree[self].need_split = (totnum > max_nodesize);
                tree[self].is_partnode = (!tree[self].need_split && tree[tree[self].parent].need_split);
            });
        });

        treecount2.assign(
            device, [this](pair<gpu_uint, gpu_uint> treerange, resource<array<Tuint, 3>>& node_memory_request) {
                gpu_uint need_child_nodes = 0;
                gpu_uint need_split_nodes = 0;
                gpu_uint partnode_nodes = 0;

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
                    need_split_nodes += (gpu_uint)tree[self].need_split;
                    partnode_nodes += (gpu_uint)tree[self].is_partnode;
                });

                need_child_nodes = work_group_reduce_add(need_child_nodes, local_size());
                need_split_nodes = work_group_reduce_add(need_split_nodes, local_size());
                partnode_nodes = work_group_reduce_add(partnode_nodes, local_size());
                gpu_if(local_id() == 0)
                {
                    // Writing memory requirements for this workgroup.
                    node_memory_request[group_id()] = { need_child_nodes, need_split_nodes, partnode_nodes };
                }
            });

        this->node_memory_request.assign(device, treecount2.num_groups());

        treecount3.assign(
            device,
            [this](pair<gpu_uint, gpu_uint> treerange) {
                gpu_uint next_child_index = this->node_memory_request[group_id()][0];
                gpu_uint next_split_index = this->node_memory_request[group_id()][1];
                gpu_uint next_partnode_index = this->node_memory_request[group_id()][2];

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

                    auto get_index = [&](gpu_bool cnd, gpu_uint& next_index) {
                        gpu_uint ret;
                        vector<gpu_uint> b = ballot(cnd, local_size());
                        gpu_uint l = local_id();
                        for (gpu_uint bt : b)
                        {
                            ret = cond(l < 32u, next_index + popcount(bt & ((1u << l) - 1)), ret);
                            next_index += popcount(bt);
                            l -= 32;
                        }
                        return ret;
                    };

                    gpu_uint child_index = get_index(have_child, next_child_index);
                    gpu_uint split_index = get_index(tree[self].need_split, next_split_index);
                    gpu_uint partnode_index = get_index(tree[self].is_partnode, next_partnode_index);

                    tree[self].first_child = cond(have_child, child_index * 2, 0u);

                    gpu_if(have_child)
                    {
                        for (Tuint childnum : { 0, 1 })
                        {
                            tree[tree[self].first_child + childnum].parent = self;
                        }
                    }
                    gpu_if(tree[self].need_split)
                    {
                        split_indices[split_index] = self;
                    }
                    gpu_if(tree[self].is_partnode)
                    {
                        partnode_indices[partnode_index] = self;
                    }
                });
            },
            this->treecount2.local_size(), // thread numbers must be the same as in treecount2.
            this->treecount2.global_size());

        treecount4.assign(device,
                          [this](const resource<pair<signature_t, Tuint>>& particles,
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

        make_IC();

        movefunc.assign(device, [this](gpu_T dt) {
            gpu_for_global(0, x.size(), [&](gpu_uint k) {
                Vector<gpu_T, 3> old_x = x[k];

                x[k] += Vector<gpu_T, 3>(v[k]) * dt;

                gpu_if(x[k].cwiseAbs().maxCoeff() >= halflen)
                {
                    x[k] = old_x;
                    v[k] = { 0, 0, 0 };
                }
                gpu_assert(isfinite(x[k].squaredNorm()));
            });
        });

        kick.assign(device, [this](gpu_T dt) {
            gpu_for_global(0, v.size(), [&](gpu_uint k) {
                v[k] += force[k] * dt;
                gpu_assert(isfinite(v[k].squaredNorm()));
            });
        });

        sort1func.assign(device, [this](resource<pair<signature_t, Tuint>>& plist, gpu_uint size) {
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

        verify.assign(device,
                      [this](
#if CALC_POTENTIAL
                          gather_add<double>& poterr,
#endif
                          gpu_uint pnum) -> gather_add<double> {
                          gpu_double ret = 0;
#if CALC_POTENTIAL
                          poterr = 0;
#endif
                          gpu_for_global(0, pnum, [&](gpu_uint p) {
                              gpu_uint a = gpu_uint(gpu_uint64(x.size()) * p / pnum);
                              Vector<gpu_double, 3> F = { 0, 0, 0 };
                              gpu_double P = 0;
                              gpu_for(0, x.size(), [&](gpu_uint b) {
                                  Vector<gpu_double, 3> dist = (x[b] - x[a]).template cast<gpu_double>();
                                  gpu_if(a != b)
                                  {
                                      F += mass[b] * dist * pow<-3, 2>(dist.squaredNorm() + 1E-20);
                                      P += -mass[b] * pow<-1, 2>(dist.squaredNorm() + 1E-20);
                                  }
                              });
                              ret += (force[a].template cast<gpu_double>() - F).squaredNorm();
#if CALC_POTENTIAL
                              poterr += pow2(potential[a] - P);
#endif
                          });
                          return ret;
                      });
    }
};

template<class T, unsigned int max_multipole>
struct cosmos : public cosmos_base<T>
{
    using gpu_T = typename gettype<T>::gpu;
    using cosmos_base<T>::vdata;
    using cosmos_base<T>::x;
    using cosmos_base<T>::v;
    using cosmos_base<T>::mass;
    using cosmos_base<T>::potential;
    using cosmos_base<T>::plist1;
    using cosmos_base<T>::plist2;
    using cosmos_base<T>::force;
    using cosmos_base<T>::tmps;

    goopax_device device;

    buffer<multipole<T, max_multipole>> matter_tree;
    buffer<multipole<T, max_multipole>> force_tree;

    array<array<kernel<void(pair<Tuint, Tuint> treerange, T scale)>, 2>, 3> upwards;

    kernel<void(pair<Tuint, Tuint> split_index_range)> downwards2;

    array<kernel<void(pair<Tuint, Tuint> partnode_index_range, T scale)>, 3> handle_particles_from_multipole;

    kernel<void(pair<Tuint, Tuint> partnode_index_range)> handle_particles_direct;

    void print(auto& tree, pair<Tuint, Tuint> treerange)
    {
        const_buffer_map map(tree);
        const_buffer_map matter(matter_tree);
        const_buffer_map surrounding(this->surrounding_buf);
        for (uint k = treerange.first; k < treerange.second; ++k)
        {
            cout << k << ": " << map[k] << ". matter: " << matter[k] << ". surrounding:";
            for (uint s = 0; s < this->num_surrounding(); ++s)
            {
                cout << " " << surrounding[k * this->num_surrounding() + s];
            }
            cout << endl;
        }
    }

    void compute_force() override
    {
#if WITH_TIMINGS
        x.get_device().wait_all();
        auto t1 = steady_clock::now();
#endif

        {
            Tdouble scale = pow(2.0, 1.0 / 3 * this->treeranges.size()) / halflen;

            for (Tuint depth = this->treeranges.size() - 1; depth != Tuint(-1); --depth)
            {
                upwards[modulo((int)depth, 3)][depth == this->treeranges.size() - 1](this->treeranges[depth], scale)
                    .wait();

                scale *= pow<-1, 3>(2.0);
            }
        }

#if WITH_TIMINGS
        x.get_device().wait_all();
        auto t2 = steady_clock::now();
#endif

        for (uint k = 0; k < this->num_particle_calls; ++k)
        {
            handle_particles_direct(
                { Tuint(size_t(k) * this->partnode_index_ranges.back().second / this->num_particle_calls),
                  Tuint(size_t((k + 1) * this->partnode_index_ranges.back().second / this->num_particle_calls)) });
        }

#if WITH_TIMINGS
        auto t3 = steady_clock::now();
#endif

#if WITH_TIMINGS
        chrono::duration<double> time_downwards = 0s;
        chrono::duration<double> time_hp = 0s;
#endif

        {
            Tdouble scale = 1.0 / halflen * pow<1, 3>(2.0);

            for (uint depth = 1; depth < this->treeranges.size(); ++depth)
            {
                scale *= pow<1, 3>(2.0);

#if WITH_TIMINGS
                x.get_device().wait_all();
                auto d0 = steady_clock::now();
#endif

                downwards2(this->split_index_ranges[depth - 1]).wait();

#if WITH_TIMINGS
                x.get_device().wait_all();
                auto d1 = steady_clock::now();
#endif

                handle_particles_from_multipole[modulo((int)depth, 3)](this->partnode_index_ranges[depth], scale)
                    .wait();
                // cout << "partnode_index_range[" << depth << "]=" << partnode_index_ranges[depth] << endl;

#if WITH_TIMINGS
                x.get_device().wait_all();
                auto d2 = steady_clock::now();

                time_downwards += (d1 - d0);
                time_hp += (d2 - d1);
#endif
            }
        }

#if WITH_TIMINGS
        x.get_device().wait_all();
        auto t4 = steady_clock::now();

        cout << "treecount: " << duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << endl;
        cout << "upwards: " << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << endl;
        cout << "handle_particles (direct): " << duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " ms"
             << endl;
        cout << "downwards: " << duration_cast<std::chrono::milliseconds>(time_downwards).count() << " ms" << endl;
        cout << "handle_particles (multipole): " << duration_cast<std::chrono::milliseconds>(time_hp).count() << " ms"
             << endl;
#endif
    }

    cosmos(goopax_device device0, Tsize_t N, size_t max_treesize, Tdouble max_distfac, unsigned int max_nodesize)
        : cosmos_base<T>(device0, N, max_treesize, max_distfac, max_nodesize)
        , device(device0)
        , matter_tree(device, this->max_treesize)
        , force_tree(device, this->max_treesize)
    {
        matter_tree.fill(zero, 0, 4);
        force_tree.fill(zero, 0, 4);

        for (unsigned int mod3 = 0; mod3 < 3; ++mod3)
        {

            for (bool is_bottom : { false, true })
            {
                upwards[mod3][is_bottom].assign(
                    device, [is_bottom, mod3, this](pair<gpu_uint, gpu_uint> treerange, gpu_T scale) {
                        gpu_for_global(treerange.first, treerange.second, [&](gpu_uint t) {
                            const gpu_bool is_pnode = (is_bottom || this->tree[t].first_child == 0);

                            multipole<gpu_T, max_multipole> Msum_r = zero;

                            gpu_if(is_pnode)
                            {
                                gpu_for(this->tree[t].pbegin, this->tree[t].pend, [&](gpu_uint p) {
                                    Msum_r += multipole<gpu_T, max_multipole>::from_particle(
                                        (rot(x[p], mod3) - this->tree[t].rcenter) * scale, mass[p]);
                                });
                            }
                            gpu_else
                            {
                                for (int child = 0; child < 2; ++child)
                                {
                                    const gpu_uint child_id = this->tree[t].first_child + child;
                                    const multipole<gpu_T, max_multipole> Mcr = matter_tree[child_id];
                                    Vector<int, 3> shift_r = { (1 - (2 * child)), 0, 0 };
                                    multipole<gpu_T, max_multipole> Mr = Mcr.rot(-1)
                                                                             .scale_ext(pow<-1, 3>(2.0))
                                                                             .shift_ext(shift_r.template cast<gpu_T>());
                                    Msum_r += Mr;
                                }
                            }

                            matter_tree[t] = Msum_r;
                        });
                    });
            }
            handle_particles_from_multipole[mod3].assign(
                device, [mod3, this](pair<gpu_uint, gpu_uint> partnode_index_range, gpu_T scale) {
                    gpu_for_group(partnode_index_range.first, partnode_index_range.second, [&](gpu_uint partnode_i) {
                        gpu_uint self = this->partnode_indices[partnode_i];
                        gpu_assert(this->tree[self].is_partnode);
                        gpu_assert(!this->tree[self].need_split && this->tree[this->tree[self].parent].need_split);

                        gpu_for_local(this->tree[self].pbegin, this->tree[self].pend, [&](gpu_uint pa) {
                            gpu_T halflen = pow<1, 3>(2.f) / scale;
                            Vector<gpu_T, 3> halflen3 = vdata.make_real({ 1, 1, 1 }, halflen);
                            for (uint k = 0; k < 3; ++k)
                            {
                                gpu_assert(abs((rot(x[pa], mod3) - this->tree[self].rcenter)[k])
                                           < halflen3[k] * 1.001f);
                            }

                            Vector<gpu_T, 3> F =
                                rot(force_tree[self].calc_force((rot(x[pa], mod3) - this->tree[self].rcenter) * scale),
                                    -(Tint)mod3)
                                * pow2(scale);
#if CALC_POTENTIAL
                            gpu_T P = force_tree[self].calc_loc_potential((rot(x[pa], mod3) - this->tree[self].rcenter)
                                                                          * scale)
                                      * scale;
#endif

                            force[pa] += F;
                            gpu_assert(isfinite(force[pa].squaredNorm()));
#if CALC_POTENTIAL
                            potential[pa] += P;
#endif
                        });
                    });
                });

            handle_particles_direct.assign(device, [this](pair<gpu_uint, gpu_uint> partnode_index_range) {
                gpu_for_group(partnode_index_range.first, partnode_index_range.second, [&](gpu_uint partnode_i) {
                    gpu_uint self = this->partnode_indices[partnode_i];
                    gpu_assert(this->tree[self].is_partnode);
                    gpu_assert(!this->tree[self].need_split && this->tree[this->tree[self].parent].need_split);

                    local_mem<Tuint> other_indices(this->max_nodesize);

                    gpu_for_local(this->tree[self].pbegin, this->tree[self].pend, [&](gpu_uint p) {
                        other_indices[p - this->tree[self].pbegin] = p;
                    });
                    gpu_uint num_other = this->tree[self].pend - this->tree[self].pbegin;

                    gpu_for_local(0, intceil(this->num_surrounding(), local_size()), [&](gpu_uint si) {
                        gpu_uint num = 0;
                        gpu_uint other;
                        gpu_if(si < this->num_surrounding())
                        {
                            other = this->surrounding_link(self, si);
                            num = this->tree[other].pend - this->tree[other].pbegin;
                        }
                        num_other += work_group_scan_exclusive_add(num);
                        gpu_if(si < this->num_surrounding())
                        {
                            gpu_for(this->tree[other].pbegin, this->tree[other].pend, [&](gpu_uint p) {
                                other_indices[num_other++] = p;
                            });
                        }
                        num_other = shuffle(num_other, local_size() - 1, local_size());
                    });

                    local_barrier(memory::threadgroup);

                    auto run = [&](gpu_uint pa0, uint par, uint ls) {
                        gpu_uint gid = local_id() / ls;
                        // uint ng = local_size() / ls;

                        gpu_uint lid = local_id() % ls;
                        // gpu_uint pa0 = begin + par * gid;

                        {
                            vector<Vector<gpu_T, 3>> F(par, { 0, 0, 0 });
                            vector<gpu_T> P(par, 0);

                            gpu_for(lid, num_other, ls, [&](gpu_uint i) {
                                gpu_uint pb = other_indices[i];
                                for (uint k = 0; k < par; ++k)
                                {
                                    gpu_uint pa = pa0 + k;
                                    gpu_if(pa != pb)
                                    {
                                        const Vector<gpu_T, 3> dist = x[pb] - x[pa];
                                        F[k] += dist
                                                * (mass[pb] * pow<-1, 2>(dist.squaredNorm() + 1E-20f)
                                                   * pow2(pow<-1, 2>(dist.squaredNorm() + 1E-20f)));

                                        P[k] += -(mass[pb] * pow<-1, 2>(dist.squaredNorm() + 1E-20f));
                                    }
                                }
                            });

                            for (uint k = 0; k < par; ++k)
                            {
                                F[k] = work_group_reduce_add(F[k], ls);
#if CALC_POTENTIAL
                                P[k] = work_group_reduce_add(P[k], ls);
#endif
                            }
                            gpu_if(lid == 0)
                            {
                                for (uint k = 0; k < par; ++k)
                                {
                                    gpu_uint pa = pa0 + k;
                                    force[pa] = F[k];
                                    gpu_assert(isfinite(force[pa].squaredNorm()));
#if CALC_POTENTIAL
                                    potential[pa] = P[k];
#endif
                                }
                            }
                        }
                    };

                    gpu_uint pbegin = this->tree[self].pbegin;
                    gpu_uint pend = this->tree[self].pend;

                    auto run_loop = [&](Tuint par, Tuint ls) {
                        gpu_uint gid = local_id() / ls;
                        Tuint stepsize = par * local_size() / ls;
                        gpu_for(pbegin + gid * par, pend - (pend - pbegin) % stepsize, stepsize, [&](gpu_uint pa) {
                            run(pa, par, ls);
                        });
                        pbegin = pend - (pend - pbegin) % stepsize;
                    };

                    // run_loop(16, 1);
                    run_loop(4, 1);
                    // run_loop(4, local_size() / 16);
                    run_loop(4, local_size() / 8);
                    run_loop(4, local_size() / 4);
                    run_loop(4, local_size() / 2);
                    run_loop(4, local_size());
                    run_loop(2, local_size());
                    run_loop(1, local_size());

                    local_barrier(memory::threadgroup);
                });
            });
        }

        this->num_particle_calls = 1;
        {
            size_t cachesize = device.cache_size();
            cout << "cachesize=" << (cachesize >> 20) << " MB" << endl;
            if (cachesize != 0)
            {
                this->num_particle_calls = this->plist1.size() * 16 / cachesize + 1;
                cout << "setting num_particle_calls=" << this->num_particle_calls << endl;
            }
        }

        downwards2.assign(device, [this](pair<gpu_uint, gpu_uint> split_index_range) {
            gpu_for_global(split_index_range.first, split_index_range.second, [&](gpu_uint split_i) {
                gpu_uint parent = this->split_indices[split_i];

                gpu_assert(this->tree[parent].need_split);

                for (int c = 0; c < 2; ++c)
                {
                    Vector<Tdouble, 3> shiftvec = { 0, 0, pow<1, 3>(2.0) * (2 * c - 1) };
                    gpu_uint self = this->tree[parent].first_child + c;
                    multipole<gpu_T, max_multipole> new_multipole =
                        force_tree[parent].rot().scale_loc(pow<1, 3>(2.0)).shift_loc(shiftvec.cast<gpu_T>());

                    Tuint count = 0;
                    set<unsigned int> handled;
                    for (Tuint parent_si = 0; parent_si < this->num_surrounding() + 1; ++parent_si)
                    {
                        Vector<Tint, 3> unclevec;
                        if (parent_si < this->num_surrounding())
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
                                gpu_uint cousin = this->tree[(parent_si == this->num_surrounding()
                                                                  ? parent
                                                                  : this->surrounding_link(parent, parent_si))]
                                                      .first_child
                                                  + c2;
                                new_multipole += matter_tree[cousin].makelocal(
                                    (-vdata.make_real(cousinvec, 2 * pow<1, 3>(2.0))).template cast<gpu_T>());
                                ++count;
                            }
                            // cout << endl;
                        }
                    }
                    for (uint si = 0; si < this->num_surrounding(); ++si)
                    {
                        assert(handled.contains(vdata.make_index(vdata.vicinity_vec[si])));
                    }

                    assert(count == this->num_surrounding() + 1);

                    force_tree[self] = new_multipole;
                }
            });
        });
    }
};
