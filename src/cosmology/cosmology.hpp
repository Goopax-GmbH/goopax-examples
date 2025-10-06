#define WITH_TIMINGS 1

#if __has_include(<opencv2/opencv.hpp>)
#include <opencv2/opencv.hpp>
#define WITH_OPENCV 1
#else
#define WITH_OPENCV 0
#endif

#include <SDL3/SDL_main.h>
#include <goopax_draw/particle.hpp>
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

#define MULTIPOLE_ORDER 4
constexpr unsigned min_tree_depth = 10;

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

template<ostream_type S, typename A, typename B>
S& operator<<(S& s, const pair<A, B>& p);

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

template<ostream_type S, typename A, typename B>
S& operator<<(S& s, const pair<A, B>& p)
{
    return s << "[" << p.first << "," << p.second << "]";
}

using signature_t = Tuint64_t;
using gpu_signature_t = typename make_gpu<signature_t>::type;

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

// #include "radix_sort.hpp"
const float top_halflen = 4;
PARAMOPT<Tuint> MAX_DEPTH("max_depth", 64);

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

auto get_index(gpu_bool cnd, auto& next_index, gpu_uint ret = {})
{
    // gpu_uint ret;
    vector<gpu_uint> b = ballot(cnd, local_size());
    gpu_uint l = local_id();
    for (gpu_uint bt : b)
    {
        ret = cond(l < 32u, next_index + popcount(bt & ((1u << l) - 1)), ret);
        next_index += popcount(bt);
        l -= 32;
    }
    return ret;
}

template<typename T>
struct scratch_treenode
{
    using bool_type = typename change_gpu_mode<bool, T>::type;
    using uint_type = typename change_gpu_mode<unsigned int, T>::type;
    using uint8_type = typename change_gpu_mode<uint8_t, T>::type;

    uint_type pbegin;
    uint_type pend;
    bool_type need_split;
    bool_type is_partnode;
    bool_type has_children;
    uint8_type depth;

    static scratch_treenode zero()
    {
        return { 0, 0, false, false, false, 0 };
    }

    template<class STREAM>
    friend STREAM& operator<<(STREAM& s, const scratch_treenode& n)
    {
        s << "[pbegin=" << n.pbegin << ", pend=" << n.pend << ", need_split=" << n.need_split
          << ", is_partnode=" << n.is_partnode << ", has_children=" << n.has_children
          << ", depth=" << static_cast<uint_type>(n.depth) << "]";
        return s;
    }
};
GOOPAX_PREPARE_STRUCT(scratch_treenode);

template<class T = Tfloat>
struct treenode : public scratch_treenode<T>
{
    using goopax_struct_type = T;
    template<typename X>
    using goopax_struct_changetype = treenode<typename goopax_struct_changetype<T, X>::type>;
    using uint_type = typename change_gpu_mode<unsigned int, T>::type;

    array<uint_type, 2> children;
    uint_type parent;
    Vector<T, 3> rcenter;
    typename change_gpu_mode<signature_t, T>::type signature;

    static treenode zero()
    {
        return reinterpret<treenode>(static_cast<array<uint_type, get_size<treenode>::value / 4>>(::zero));
    }

    template<class STREAM>
    friend STREAM& operator<<(STREAM& s, const treenode& n)
    {
        s << "[" << static_cast<const scratch_treenode<T>&>(n) << ", children=" << n.children << ", parent=" << n.parent
          << ", rcenter=" << n.rcenter << ", signature=" << hex << n.signature << dec << "]";
        return s;
    }
};

gpu_type<Tuint*> get_children_p(gpu_type<treenode<>*> p)
{
    return reinterpret<gpu_type<Tuint*>>(p)
           + (&reinterpret<treenode<>*>(nullptr)->children[0]
              - reinterpret_cast<Tuint*>(reinterpret<treenode<>*>(nullptr)));
}

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
            sig_t s = sig_t(abs(x[k]) * (1.0 / (top_halflen * pow(2.0, Tdouble(2 - k) / 3)) * (1 << (depth[k] - 1))));
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
            sig_t s = sig_t(
                abs(x[k]) * static_cast<T>(1.0 / (top_halflen * pow(2.0, Tdouble(2 - k) / 3)) * (1 << (depth[k] - 1))));
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
    // Vector<Tuint, 3> depth = { (max_depthbits + 2) / 3, (max_depthbits + 1) / 3, (max_depthbits) / 3 };

    sig_t ret = calc_sig_fast<T>(x, signature_t()) >> (get_size<sig_t>::value * 8 - max_depthbits);

    return ret;
}

void multi_level_loop(resource<pair<Tuint, Tuint>> map, gpu_uint tid, Tuint nt, std::function<void(gpu_uint)> func)
{
    auto node_p = map.cbegin();
    gpu_uint self = tid + node_p->first;

    gpu_while(node_p->second != 0)
    {
        gpu_while(self < node_p->second)
        {
            func(self);

            self += nt;
        }
        self += node_p[1].first - node_p[0].second;
        ++node_p;
    }
}

struct vicinity_data
{
    vector<Vector<Tint, 3>> local_vec;
    set<Tuint> local_indices;
    vector<Vector<Tint, 3>> vicinity_vec;
    const double max_distfac;

    static Tint make_index(Vector<Tint, 3> r)
    {
        return r[2] * 65536 + r[1] * 256 + r[0];
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
                    // Vector<Tdouble, 3> center = make_real(ac, Tdouble(1.0));
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

        // cout << "local_vec=" << local_vec << endl;
        // cout << "vicinity_vec=" << vicinity_vec << endl;

        assert(max_e > maxvec[0]);
        assert(max_e > maxvec[1]);
        assert(max_e > maxvec[2]);
    }
};

template<typename U = Tuint>
struct Node_memory_request
{
    U split;
    U withchild;
    U partnode_withchild;
    U partnode_leaf;
    U leaf;

    static Node_memory_request zero()
    {
        return { 0, 0, 0, 0, 0 };
    }
    template<typename T>
    Node_memory_request<T> cast() const
    {
        return { .split = split,
                 .withchild = withchild,
                 .partnode_withchild = partnode_withchild,
                 .partnode_leaf = partnode_leaf,
                 .leaf = leaf };
    }

    Node_memory_request& operator+=(const Node_memory_request& b)
    {
        split += b.split;
        withchild += b.withchild;
        partnode_withchild += b.partnode_withchild;
        partnode_leaf += b.partnode_leaf;
        leaf += b.leaf;
        return *this;
    }

    template<ostream_type S>
    friend S& operator<<(S& s, const Node_memory_request& n)
    {
        return s << "[split=" << n.split << ", withchild=" << n.withchild
                 << ", partnode_withchild=" << n.partnode_withchild << ", partnode_leaf=" << n.partnode_leaf
                 << ", leaf=" << n.leaf << "]";
    }
};
GOOPAX_PREPARE_STRUCT(Node_memory_request);

template<typename U>
Node_memory_request<U> operator+(Node_memory_request<U> a, const Node_memory_request<U>& b)
{
    return a += b;
}

template<class T, unsigned int max_multipole>
struct cosmos
{
#if MULTIPOLE_ORDER == 1
    using matter_multipole = multipole<T, T>;
    using force_multipole = multipole<T, T>;
#elif MULTIPOLE_ORDER == 2
    using matter_multipole = multipole<T, T, T>;
    using force_multipole = multipole<T, T, T>;
#elif MULTIPOLE_ORDER == 3
    using matter_multipole = multipole<T, T, T, T>;
    using force_multipole = multipole<T, T, T, T>;
#elif MULTIPOLE_ORDER == 4
    using matter_multipole = multipole<T, T, T, T, T>;
    using force_multipole = multipole<T, T, T, T, T>;
#else
#error
#endif
    using gpu_matter_multipole = typename make_gpu<matter_multipole>::type;
    using gpu_force_multipole = typename make_gpu<force_multipole>::type;

    goopax_device device;
    const Tuint num_particles;
    using gpu_T = typename make_gpu<T>::type;
    buffer<Vector<T, 3>> x;
    buffer<Vector<T, 3>> v;
#if CALC_POTENTIAL
    buffer<T> potential;
#endif
    buffer<T> mass;
    buffer<Vector<T, 3>> force;
    buffer<T> tmps; // FIXME: Reduce memory.

    const vicinity_data vdata;

    const size_t max_treesize;
    const unsigned int max_nodesize;
    // Tuint num_particle_calls;

    buffer<treenode<T>> tree;

    // buffer<Tuint> split_indices;
    // buffer<Tuint> partnode_indices;
    buffer<Tuint> surrounding_buf;

    vector<pair<Tuint, Tuint>> treeranges;
    vector<array<Tuint, 3>> treeranges_withchild_nochild;

    vector<pair<Tuint, Tuint>> split_ranges;
    vector<pair<Tuint, Tuint>> withchild_ranges;
    vector<pair<Tuint, Tuint>> partnode_withchild_ranges;
    vector<pair<Tuint, Tuint>> partnode_leaf_ranges;
    vector<pair<Tuint, Tuint>> leaf_ranges;

    vector<pair<Tuint, Tuint>> partnode_ranges;
    vector<pair<Tuint, Tuint>> partnode_ranges_packed;
    buffer<pair<Tuint, Tuint>> partnode_ranges_packed_buf;

    vector<pair<Tuint, Tuint>> all_leaf_ranges_packed;
    buffer<pair<Tuint, Tuint>> all_leaf_ranges_packed_buf;

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

    buffer<Node_memory_request<>> node_memory_request;

    void print(auto& tree, pair<Tuint, Tuint> treerange, bool with_surrounding = false)
    {
        const_buffer_map map(tree);
        const_buffer_map surrounding(this->surrounding_buf);
        // const_buffer_map matter(multipole_tree);
        // const_buffer_map surrounding(this->surrounding_buf);
        for (uint k = treerange.first; k < treerange.second; ++k)
        {
            cout << k << ": " << map[k];
            if (with_surrounding)
            {
                cout << ". surrounding:";
                for (uint s = 0; s < this->num_surrounding(); ++s)
                {
                    cout << " " << surrounding[k * this->num_surrounding() + s];
                }
            }
            cout << endl;
        }
    }

    kernel<void(pair<Tuint, Tuint> treerange)> save_old_tree_data;

    kernel<void(pair<Tuint, Tuint> treerange, Tuint parent_begin)> treecount1;

    kernel<void(pair<Tuint, Tuint> treerange, buffer<Node_memory_request<>>& node_memory_request)> treecount2;

    kernel<void(pair<Tuint, Tuint> treerange,
                array<Tuint, 3> treeranges_parent_withchild_nochild,
                T halflen,
                Tuint8_t depth,
                Node_memory_request<> memory_offset)>
        treecount3;

    kernel<void(pair<Tuint, Tuint> treerange_with_child, Tuint treebegin_next, T halflen_sublevel)>
        treecount4_first_depth;

    array<kernel<void(pair<Tuint, Tuint> treerange_with_child,
                      Tuint treebegin_next,
                      Tuint8_t depth_sublevel,
                      T halflen_sublevel,
                      // buffer<treenode<T>>& dest_tree,
                      buffer<Vector<T, 3>>& x,
                      buffer<Vector<T, 3>>& v,
                      buffer<T>& mass)>,
          3>
        treecount4;

    kernel<void(pair<Tuint, Tuint> treerange_leaf)> make_tree_clear_leaf_childs;

    kernel<void(pair<Tuint, Tuint> treerange_parent, pair<Tuint, Tuint> treerange, T halflen)> make_surrounding;
    kernel<void(pair<Tuint, Tuint> treerange_parent, pair<Tuint, Tuint> treerange, T halflen)>
        make_surrounding_with_scratch;

    array<kernel<void(
              const buffer<Vector<T, 3>>& x, const buffer<T>& mass, buffer<Vector<T, 3>>& force, buffer<T>& potential)>,
          2>
        handle_particles_direct2;

    kernel<void(T dt, buffer<Vector<T, 3>>& v, buffer<Vector<T, 3>>& x)> movefunc;

    kernel<void(T dt, const buffer<Vector<T, 3>>& force, buffer<Vector<T, 3>>& v)> kick;

    kernel<void(const buffer<Vector<T, 3>>& in, buffer<Vector<T, 3>>& out)> apply_vec2;

    kernel<void(const buffer<T>& in, buffer<T>& out)> apply_scalar2;

    kernel<double(const buffer<Vector<T, 3>>& x,
                  const buffer<Vector<T, 3>>& force,
                  const buffer<T>& mass,
                  const buffer<T>& potential,
#if CALC_POTENTIAL
                  goopax_future<double>& poterr,
#endif
                  Tuint pnum)>
        verify;

    // radix_sort<pair<signature_t, Tuint>, signature_t> Radix;

    void make_tree_base()
    {
        cout << "make_tree_base" << endl;
        surrounding_buf.fill(0, 0, 4 * num_surrounding());
        {
            buffer_map tree(this->tree, 0, 2 + (1 << (min_tree_depth + 1)));
            tree[2].parent = 0;
            tree[3].parent = 0;

            double level_halflen = top_halflen;
            Tuint offset = 3;
            for (Tuint depth = 0; depth <= min_tree_depth; ++depth)
            {
                pair<Tuint, Tuint> treerange = { offset, offset + (1 << depth) };
                treeranges.push_back(treerange);
                split_ranges.push_back(treerange);
                withchild_ranges.push_back({ treerange.second, treerange.second });
                partnode_withchild_ranges.push_back({ treerange.second, treerange.second });
                partnode_leaf_ranges.push_back({ treerange.second, treerange.second });
                leaf_ranges.push_back({ treerange.second, treerange.second });
                partnode_ranges.push_back({ treerange.second, treerange.second });
                treeranges_withchild_nochild.push_back({ treerange.first, treerange.second, treerange.second });

                // cout << "depth=" << depth << ", treerange=" << treerange << endl;

                for (Tuint self = treerange.first; self < treerange.second; ++self)
                {
                    tree[self].need_split = true;
                    tree[self].is_partnode = false;
                    tree[self].children = { 2 * self - 2, 2 * self - 1 };
                    tree[self].parent = self == 3 ? Tuint(0) : (self + 2) / 2;

                    Tuint parent = tree[self].parent;
                    Tuint c = self % 2;
                    // cout << "self=" << self << ", parent=" << parent << ", c=" << c << endl;
                    // cout << "now: node[" << self << "]=" << tree[self] << endl;

                    Vector<T, 3> rcenter = tree[parent].rcenter;
                    if (self != 3)
                    {
                        rcenter[0] += (c == 0 ? -level_halflen : level_halflen);
                        tree[self].signature = tree[parent].signature * 2 + c;
                    }
                    tree[self].rcenter = rot(rcenter);
                    tree[self].depth = depth;

                    tree[self].pbegin = 0;
                    tree[self].pend = 0;
                }

                level_halflen *= pow(2.0, -1.0 / 3);
                offset = treerange.second;
            }
        }

        // cout << "root:\n";
        // print(this->tree, { 3, 4 });

        for (Tuint depth = 1; depth < min_tree_depth; ++depth)
        {
            make_surrounding(treeranges[depth - 1], treeranges[depth], top_halflen * pow(2.0, -1.0 / 3 * depth)).wait();
        }
    }

    void make_initial_tree()
    {
        make_tree_base();
        make_tree();

        cout << "Assigning particles to some initial nodes." << endl;

        {
            kernel assign(device, [&]() {
                gpu_for_global(
                    partnode_leaf_ranges.back().first, partnode_leaf_ranges.back().second, [&](gpu_uint self) {
                        tree[self].pbegin = static_cast<gpu_uint>(
                            gpu_size_t(self - partnode_leaf_ranges.back().first) * x.size()
                            / (partnode_leaf_ranges.back().second - partnode_leaf_ranges.back().first));
                        tree[self].pend = static_cast<gpu_uint>(
                            gpu_size_t(self - partnode_leaf_ranges.back().first + 1) * x.size()
                            / (partnode_leaf_ranges.back().second - partnode_leaf_ranges.back().first));
                    });
            });

            cout << "Calling assign. partnode_leaf_ranges.back()=" << partnode_leaf_ranges.back() << endl;
            assign();
        }

        update_tree(false);

        cout << "Tree initialized." << endl;
    }

    void make_tree()
    {
        // cout << "make_tree" << endl;

#if WITH_TIMINGS
        vector<chrono::duration<double>> treetime(7, 0s);
        device.wait_all();
        auto t0 = steady_clock::now();
#endif

        save_old_tree_data({ 0, treeranges.back().second });

        this->treeranges.resize(min_tree_depth);
        this->split_ranges.resize(min_tree_depth);
        this->withchild_ranges.resize(min_tree_depth);
        this->partnode_withchild_ranges.resize(min_tree_depth);
        this->partnode_leaf_ranges.resize(min_tree_depth);
        this->leaf_ranges.resize(min_tree_depth);
        treeranges_withchild_nochild.resize(min_tree_depth);

        partnode_ranges.resize(min_tree_depth);
        partnode_ranges_packed.clear();
        all_leaf_ranges_packed.clear();

#if WITH_TIMINGS
        device.wait_all();
        auto t1 = steady_clock::now();
#endif

        Tuint treesize = (1u << min_tree_depth);
        Tuint treeoffset = 2 + (1 << (min_tree_depth));

#if GOOPAX_DEBUG
        scratch_tree.fill({});
        surrounding_buf.fill({}, treeoffset * num_surrounding(), surrounding_buf.size());
#endif

        scratch_tree.fill(zero, 0, 2);

        // this->scratch_tree.copy(this->tree, treesize, treeoffset, treeoffset);
#if GOOPAX_DEBUG
        scratch_tree.fill({}, treeoffset, tree.size());
#endif

        /*
          cout << "BEFORE make_tree:" << endl;
            for (uint depth = 0; depth < treeranges.size(); ++depth)
            {
                cout << "depth=" << depth << ":\n";
                print(this->tree, treeranges[depth], true);
            }
        */

        {
            double level_halflen = top_halflen * pow<-(int)min_tree_depth, 3>(2.0);
            // Tuint parent_begin = 0;

            treecount4_first_depth(
                { treeranges[min_tree_depth - 1].first, partnode_withchild_ranges[min_tree_depth - 1].second },
                treeranges[min_tree_depth - 1].second,
                top_halflen * pow(2.0, (-1 - Tint(min_tree_depth - 1)) / 3.0));

            for (Tuint depth = min_tree_depth; depth < MAX_DEPTH(); ++depth)
            {
                pair<Tuint, Tuint> treerange = { treeoffset, treeoffset + treesize };
                // cout << "depth=" << depth << ", treerange=" << treerange << endl;

                this->treeranges.push_back(treerange);

                if (depth == MAX_DEPTH() - 1)
                    break;

#if WITH_TIMINGS
                device.wait_all();
                auto g0 = steady_clock::now();
#endif

                make_surrounding_with_scratch(
                    { treeranges_withchild_nochild[depth - 1][0], treeranges_withchild_nochild[depth - 1][1] },
                    treerange,
                    level_halflen)
                    .wait();

#if WITH_TIMINGS
                device.wait_all();
                auto g1 = steady_clock::now();
#endif
                // cout << "calling treecount1. treerange=" << treerange << endl;
                this->treecount1(treerange, treeranges[depth - 1].first).wait();

                // cout << "after treecount1: scratch_tree:\n";
                // print(this->scratch_tree, treerange);

#if WITH_TIMINGS
                device.wait_all();
                auto g2 = steady_clock::now();
#endif

                this->treecount2(treerange, this->node_memory_request);

                // cout << "after treecount2: scratch_tree:\n";
                // print(this->scratch_tree, treerange);

#if WITH_TIMINGS
                device.wait_all();
                auto g5 = steady_clock::now();
#endif
                Node_memory_request<Tuint> memory_offset;
                {
                    Node_memory_request<Tuint> sum = zero;
                    {
                        buffer_map node_memory_request(this->node_memory_request);
                        for (auto& n : node_memory_request)
                        {
                            auto tmp = n;
                            n = sum;
                            sum += tmp;
                        }
                    }

                    split_ranges.push_back({ treeoffset, treeoffset + sum.split });
                    treeoffset += sum.split;

                    withchild_ranges.push_back({ treeoffset, treeoffset + sum.withchild });
                    treeoffset += sum.withchild;

                    partnode_withchild_ranges.push_back({ treeoffset, treeoffset + sum.partnode_withchild });
                    treeoffset += sum.partnode_withchild;

                    partnode_leaf_ranges.push_back({ treeoffset, treeoffset + sum.partnode_leaf });
                    treeoffset += sum.partnode_leaf;

                    leaf_ranges.push_back({ treeoffset, treeoffset + sum.leaf });
                    treeoffset += sum.leaf;

                    partnode_ranges.push_back(
                        { partnode_withchild_ranges.back().first, partnode_leaf_ranges.back().second });
                    if (partnode_withchild_ranges.back().first != partnode_leaf_ranges.back().second)
                    {
                        partnode_ranges_packed.push_back(
                            { partnode_withchild_ranges.back().first, partnode_leaf_ranges.back().second });
                    }
                    if (partnode_leaf_ranges.back().first != treerange.second)
                    {
                        all_leaf_ranges_packed.push_back({ partnode_leaf_ranges.back().first, treerange.second });
                    }

                    treesize = 2 * (sum.split + sum.withchild + sum.partnode_withchild);

                    treeranges_withchild_nochild.push_back(
                        { treerange.first, partnode_withchild_ranges.back().second, treerange.second });

                    memory_offset.split = treerange.first;
                    memory_offset.withchild = memory_offset.split + sum.split;
                    memory_offset.partnode_withchild = memory_offset.withchild + sum.withchild;
                    memory_offset.partnode_leaf = memory_offset.partnode_withchild + sum.partnode_withchild;
                    memory_offset.leaf = memory_offset.partnode_leaf + sum.partnode_leaf;

                    /*
                      cout << "treeoffset=" << treeoffset << ", treerange=" << treerange
                      << ", sum=" << sum
                      << ", memory_offset=" << memory_offset
                      << endl;
                    */
                }

                assert(treeoffset == treerange.second);

                if (treerange.second + treesize > tree.size())
                {
                    throw std::runtime_error("tree too small");
                }

                treecount3(treerange,
                           (depth == 0 ? array<Tuint, 3>{ 0, 0, 0 } : treeranges_withchild_nochild[depth - 1]),
                           level_halflen,
                           depth,
                           memory_offset);

                // cout << "after treecount3: tree:\n";
                // print(this->tree, treerange);

                make_surrounding(
                    { treeranges_withchild_nochild[depth - 1][0], treeranges_withchild_nochild[depth - 1][1] },
                    treerange,
                    level_halflen)
                    .wait();

#if WITH_TIMINGS
                device.wait_all();
                auto g6 = steady_clock::now();
#endif

                /*
                  cout << "calling treecount4. new range: " << pair{treerange.first,
                 partnode_withchild_ranges.back().second}
                  << ", treesize=" << treesize
                     << ", -> next_begin=" << treerange.first + treesize
                 << endl;
                */

                treecount4[depth % 3]({ treerange.first, partnode_withchild_ranges.back().second },
                                      treerange.second,
                                      depth + 1,
                                      top_halflen * pow(2.0, (-1 - Tint(depth)) / 3.0),
                                      x,
                                      v,
                                      mass);
#if WITH_TIMINGS
                device.wait_all();
                auto g7 = steady_clock::now();
#endif

                Vector<T, 3> boxsize;
                for (Tuint k = 0; k < 3; ++k)
                {
                    boxsize[k] = top_halflen * (pow(2.0, -Tint(depth + 2 - k) / 3 + (2.0 - k) / 3.0) + 1E-7);
                }
                // cout1 << "boxsize=" << boxsize << endl;

                // cout << "after treecount4\n";
                // print(this->tree, treerange);

                // parent_begin = treerange.first;

                if (treesize == 0)
                    break;

                level_halflen *= pow(2.0, -1.0 / 3);

#if WITH_TIMINGS
                treetime[0] += (g1 - g0);
                treetime[1] += (g2 - g1);
                // treetime[2] += (g3 - g2);
                // treetime[3] += (g4 - g3);
                treetime[4] += (g5 - g2);
                treetime[5] += (g6 - g5);
                treetime[6] += (g7 - g6);
#endif
            }

            make_tree_clear_leaf_childs(
                { treeranges_withchild_nochild.back()[1], treeranges_withchild_nochild.back()[2] });

            // cout << "tree size: " << treeoffset << " / " << tree.size() << "\nsplit_index size: " <<
            // split_index_offset
            //<< "\npartnode size: " << partnode_index_offset << endl;
        }

        partnode_ranges_packed.push_back({ 0, 0 });
        partnode_ranges_packed_buf.copy_from_host_async(
            partnode_ranges_packed.data(), 0, partnode_ranges_packed.size());

        all_leaf_ranges_packed.push_back({ 0, 0 });
        all_leaf_ranges_packed_buf.copy_from_host_async(
            all_leaf_ranges_packed.data(), 0, all_leaf_ranges_packed.size());

#if GOOPAX_DEBUG
        scratch_tree.fill({});
#endif
        force_tree.fill(zero, 0, 4);

        /*
            cout << "AFTER make_tree:" << endl;
        for (uint depth=0; depth<treeranges.size(); ++depth)
          {
            cout << "depth=" << depth << ":\n";
            print(this->tree, treeranges[depth], true);
          }
        */

        if (treeranges.size() > MAX_DEPTH())
        {
            cerr << "treerange.size()=" << treeranges.size() << " > MAX_DEPTH=" << MAX_DEPTH() << endl;
            throw std::runtime_error("MAX_DEPTH exceeded");
        }

#if WITH_TIMINGS
        device.wait_all();
        auto t2 = steady_clock::now();
        cout << "\nmake_tree:\n"
             << "  sort particles: " << duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << endl
             << "  tree: " << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << endl
             << "    make_surrounding: " << duration_cast<chrono::milliseconds>(treetime[0]) << endl
             << "    treecount1: " << duration_cast<chrono::milliseconds>(treetime[1])
             << endl
             // cout << "  fill: " << duration_cast<chrono::milliseconds>(treetime[2]) << endl;
             // cout << "  surrounding buf: " << duration_cast<chrono::milliseconds>(treetime[3]) << endl;
             << "    treecount2: " << duration_cast<chrono::milliseconds>(treetime[4]) << endl
             << "    treecount3: " << duration_cast<chrono::milliseconds>(treetime[5]) << endl
             << "    treecount4: " << duration_cast<chrono::milliseconds>(treetime[6]) << endl;
#endif
    }

    void make_IC(const char* filename = nullptr)
    {
        cout << "generating initial conditions..." << flush;
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
        cout << "ok" << endl;
    }

    void precision_test()
    {
        if (!verify.valid())
        {
            verify.assign(device,
                          [this](const resource<Vector<T, 3>>& x,
                                 const resource<Vector<T, 3>>& force,
                                 const resource<T>& mass,
                                 const resource<T>& potential,
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

        // cout << "Doing precision test" << endl;
        const Tuint np = min(x.size(), (Tuint)100);

        goopax_future<double> poterr;
        Tdouble tot = verify(x,
                             force,
                             mass,
                             potential,
#if CALC_POTENTIAL
                             poterr,
#endif
                             np)
                          .get();
        cout << "err=" << sqrt(tot / np) << ", poterr=" << sqrt(poterr.get() / np) << endl;

        static ofstream PLOT("plot");

        PLOT << sqrt(tot / np) << " " << sqrt(poterr.get() / np) << " " << endl;
    }

    buffer<matter_multipole> matter_tree;
    buffer<force_multipole> force_tree;

    buffer<Tuint> scratch;
    buffer<scratch_treenode<T>> scratch_tree;

    array<kernel<void(array<Tuint, 3> treerange, T scale, const buffer<Vector<T, 3>>& x, const buffer<T>& mass)>, 3>
        upwards;

    array<kernel<void(pair<Tuint, Tuint> split_range)>, 2> downwards2;

    array<array<kernel<void(pair<Tuint, Tuint> partnode_range,
                            T scale,
                            const buffer<Vector<T, 3>>& x,
                            buffer<Vector<T, 3>>& force,
                            buffer<T>& potential)>,
                3>,
          2>
        handle_particles_from_multipole;

    // buffer<pair<Tuint, Tuint>> scratch2;
    static constexpr unsigned int scratch_offset = 4 * sizeof(matter_multipole) / sizeof(Tuint);

    const Tuint scratch_offset_tree = scratch_offset;

    const Tuint scratch_offset_next_p = scratch_offset;
    // const Tuint scratch_offset_num_removed = scratch_offset_next_p + this->tree.size();
    const Tuint scratch_offset_particle_list = scratch_offset_next_p + this->tree.size();
    const Tuint scratch_offset_nodelink = scratch_offset_particle_list + num_particles;

    const Tuint scratch_offset_new_num_p = scratch_offset_nodelink + num_particles;

    const Tuint scratch_offset_old_p_range = scratch_offset_new_num_p + this->tree.size();
    const Tuint scratch_offset_group_sum = scratch_offset_old_p_range + 2 * this->tree.size();
    const Tuint scratch_offset_new_order = scratch_offset_nodelink;

    const Tuint scratch_offset_old_children = scratch_offset_tree;
    const Tuint scratch_offset_old_node = scratch_offset_group_sum + tree.size();

    kernel<void()> update_tree_set_nodelink;
    kernel<void(const buffer<Vector<T, 3>>& x)> update_tree_1;
    kernel<void(pair<Tuint, Tuint> leaf_range, pair<Tuint, Tuint> withchild_range)> update_tree_upwards;
    kernel<void(pair<Tuint, Tuint> treerange_parent)> update_tree_downwards;
    kernel<void()> update_tree_reorder_particles;

    void compute_force(bool with_potential)
    {
#if WITH_TIMINGS
        device.wait_all();
        auto t1 = steady_clock::now();
#endif

        {
            Tdouble scale = pow(2.0, 1.0 / 3 * this->treeranges.size()) / top_halflen;

            for (Tuint depth = this->treeranges.size() - 1; depth != Tuint(-1); --depth)
            {
                upwards[modulo((int)depth, 3)](this->treeranges_withchild_nochild[depth], scale, x, mass);

                scale *= pow<-1, 3>(2.0);
            }
        }

#if WITH_TIMINGS
        device.wait_all();
        auto t2 = steady_clock::now();
#endif

        /*
          cout << "calling handle_particles_direct." << endl;
      cout << "partnode_ranges:\n";
      for (auto& p : const_buffer_map(partnode_ranges_buf))
        {
          cout << p << endl;
        }
      cout << endl;
        */

        this->handle_particles_direct2[with_potential](x, mass, force, potential);

#if WITH_TIMINGS
        device.wait_all();
        auto t3 = steady_clock::now();
#endif

#if WITH_TIMINGS
        chrono::duration<double> time_downwards = 0s;
        chrono::duration<double> time_hp = 0s;
#endif

        {
            Tdouble scale = 1.0 / top_halflen * pow<1, 3>(2.0);

            for (uint depth = 1; depth < this->treeranges.size(); ++depth)
            {
                scale *= pow<1, 3>(2.0);

#if WITH_TIMINGS
                device.wait_all();
                auto d0 = steady_clock::now();
#endif

                downwards2[with_potential](this->split_ranges[depth - 1]);

#if WITH_TIMINGS
                device.wait_all();
                auto d1 = steady_clock::now();
#endif

                handle_particles_from_multipole[with_potential][modulo((int)depth, 3)](
                    this->partnode_ranges[depth], scale, x, force, potential);
                // cout << "partnode_index_range[" << depth << "]=" << partnode_index_ranges[depth] << endl;

#if WITH_TIMINGS
                device.wait_all();
                auto d2 = steady_clock::now();

                time_downwards += (d1 - d0);
                time_hp += (d2 - d1);
#endif
            }
        }

#if WITH_TIMINGS
        // cout << "treecount: " << duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << endl;
        cout << "\ncompute_force:\n"
             << "  upwards: " << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << endl
             << "  handle_particles (direct): " << duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " ms"
             << endl
             << "  downwards: " << duration_cast<std::chrono::milliseconds>(time_downwards).count() << " ms" << endl
             << "  handle_particles (multipole): " << duration_cast<std::chrono::milliseconds>(time_hp).count() << " ms"
             << endl;
#endif
    }

    void update_tree(bool preserve_potential)
    {
        /*
      #if GOOPAX_DEBUG
          scratch.fill({}, scratch_offset, scratch.size());
  #endif
        */

        scratch.fill(~0u, scratch_offset_next_p, scratch_offset_next_p + this->treeranges.back().second);
        scratch.fill(0xfffffffe, scratch_offset_particle_list, scratch_offset_particle_list + x.size());
        scratch.fill(0, scratch_offset_new_num_p, scratch_offset_new_num_p + 2);

#if WITH_TIMINGS
        device.wait_all();
        auto t0 = steady_clock::now();
#endif

        update_tree_set_nodelink();

        // cout << "calling update_tree_1" << endl;
        // cout << "all_leaf_ranges_packed=" << all_leaf_ranges_packed_buf << endl;
#if WITH_TIMINGS
        device.wait_all();
        auto t1 = steady_clock::now();
#endif
        update_tree_1(x);

#if WITH_TIMINGS
        device.wait_all();
        auto t2 = steady_clock::now();
#endif

        for (Tint depth = this->treeranges.size() - 1; depth != -1; --depth)
        {
            // cout << "calling upwards. depth=" << depth << ", partnode_range: " << this->partnode_ranges[depth]
            //<< ", si range: " << this->split_ranges[depth]
            //<< endl;
            update_tree_upwards({ this->partnode_leaf_ranges[depth].first, treeranges[depth].second },
                                { this->treeranges[depth].first, this->partnode_withchild_ranges[depth].second });
        }

#if WITH_TIMINGS
        device.wait_all();
        auto t3 = steady_clock::now();
#endif

        for (Tuint depth = 1; depth < this->treeranges.size(); ++depth)
        {
            // cout << "calling update_tree_downwards(" << this->treeranges[depth] << "), depth=" << depth << endl;
            update_tree_downwards(
                { treeranges_withchild_nochild[depth - 1][0], treeranges_withchild_nochild[depth - 1][1] });

            // cout << "after downwards: tree:\n";
            // print(this->tree, treeranges[depth]);
        }
#if WITH_TIMINGS
        device.wait_all();
        auto t4 = steady_clock::now();
#endif

        // cout << "plist:" << endl;
        // for (auto& p : const_buffer_map(plist1))
        //{
        // cout << p << endl;
        // }
        // cout << endl;

        // cout << "calling update_tree_reorder_particles. leaf_ra

        update_tree_reorder_particles();

#if WITH_TIMINGS
        device.wait_all();
        auto t5 = steady_clock::now();
#endif

        this->apply_vec2(x, force);
        swap(x, force);
        this->apply_vec2(v, force);
        swap(v, force);
        this->apply_scalar2(mass, tmps);
        swap(mass, tmps);
        if (preserve_potential)
        {
            this->apply_scalar2(potential, tmps);
            swap(potential, tmps);
        }

#if WITH_TIMINGS
        device.wait_all();
        auto t6 = steady_clock::now();
#endif

        /*
        cout << "AFTER update_tree:" << endl;
        for (uint depth=0; depth<treeranges.size(); ++depth)
          {
            cout << "depth=" << depth << ":\n";
            print(this->tree, treeranges[depth], true);
          }
        */

#if GOOPAX_DEBUG
        scratch.fill({}, scratch_offset, scratch.size());
#endif

#if WITH_TIMINGS
        // cout << "treecount: " << duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << endl;
        cout << "\nupdate_tree:\n"
             << "  set_sig: " << duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << endl
             << "  update_tree_1: " << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << endl
             << "  upwards: " << duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " ms" << endl
             << "  downwards: " << duration_cast<std::chrono::milliseconds>(t4 - t3).count() << " ms" << endl
             << "  reorder: " << duration_cast<std::chrono::milliseconds>(t5 - t4).count() << " ms" << endl
             << "  swap_particles: " << duration_cast<std::chrono::milliseconds>(t6 - t5).count() << " ms" << endl;
#endif
    }

    template<typename V>
    void apply_func(const resource<V>& in, resource<V>& out)
    {
        auto new_order = scratch.begin() + scratch_offset_new_order;
        gpu_for_global(0, in.size(), [&](gpu_uint k) { out[k] = in[new_order[k]]; });
    }

    cosmos(goopax_device device0,
           Tsize_t N,
           size_t max_treesize0,
           Tdouble max_distfac,
           unsigned int max_nodesize0,
           backend_create_params backend_params)
        : device(device0)
        , num_particles(N)
        , x(device, N, backend_params)
        , v(device, N, backend_params)
        ,
#if CALC_POTENTIAL
        potential(device, N, backend_params)
        ,
#endif
        mass(device, N, backend_params)
        , force(device, N, backend_params)
        , tmps(device, N, backend_params)
        , vdata(max_distfac)
        , max_treesize(max_treesize0)
        , max_nodesize(max_nodesize0)
        , tree(device, max_treesize)
        //, split_indices(device, max_treesize)
        //, partnode_indices(device, max_treesize)
        , surrounding_buf(device, max_treesize * num_surrounding())
        , partnode_ranges_packed_buf(device, MAX_DEPTH() + 1)
        , all_leaf_ranges_packed_buf(device, MAX_DEPTH() + 1)
        //, Radix(device, [](auto a, auto b) { return a.first < b.first; })
        , matter_tree(device, this->max_treesize)
        , force_tree(device, this->max_treesize)
        , scratch(reinterpret<buffer<Tuint>>(matter_tree))
        , scratch_tree(reinterpret<buffer<scratch_treenode<T>>>(force_tree))
    {
        // this->scratch = reinterpret<buffer<Tuint>>(force_tree);

        static_assert(sizeof(scratch_treenode<T>) <= sizeof(force_multipole), "force_tree too small for scratch_tree");
        // this->scratch_tree = reinterpret<buffer<treenode<T>>>(force_tree);

        matter_tree.fill(zero, 0, 4);
        force_tree.fill(zero, 0, 4);

        tree.fill(zero, 0, 4);

        save_old_tree_data.assign(device, [this](pair<gpu_uint, gpu_uint> treerange) {
            auto old_p_range = reinterpret<gpu_type<pair<Tuint, Tuint>*>>(scratch.begin() + scratch_offset_old_p_range);
            auto old_children = reinterpret<gpu_type<array<Tuint, 2>*>>(scratch.begin() + scratch_offset_old_children);
            auto old_node_p = scratch.begin() + scratch_offset_old_node;

            gpu_for_global(treerange.first, treerange.second, [&](gpu_uint self) {
                old_p_range[self] = { tree[self].pbegin, tree[self].pend };
                old_children[self] = tree[self].children;
            });

            gpu_for_global(0, tree.size(), [&](gpu_uint self) {
                old_node_p[self] = cond(self < 2u + (1u << (min_tree_depth)), self, 0u);
            });
        });

        treecount1.assign(device, [this](pair<gpu_uint, gpu_uint> treerange, gpu_uint parent_begin) {
            gpu_for_global(treerange.first, treerange.second, [&](gpu_uint self) {
                gpu_uint parent = parent_begin + (self - treerange.first) / 2;

                gpu_uint totnum = scratch_tree[self].pend - scratch_tree[self].pbegin;
                for (uint si = 0; si < num_surrounding(); ++si)
                {
                    gpu_uint cousin = surrounding_link(self, si);
                    totnum += scratch_tree[cousin].pend - scratch_tree[cousin].pbegin;
                }

                // If true, force calculation wants to go further down the tree.
                scratch_tree[self].need_split = (totnum > max_nodesize);

                scratch_tree[self].is_partnode = (!scratch_tree[self].need_split && tree[parent].need_split);
            });
        });

        treecount2.assign(
            device, [this](pair<gpu_uint, gpu_uint> treerange, resource<Node_memory_request<>>& node_memory_request) {
                Node_memory_request<gpu_uint> nmr = zero;

                // Choosing loop order here in such a way that neighboring nodes in space will also be
                // neighboring nodes in memory. This loop must have the same ordering in treecount2 and
                // treecount3.
                gpu_uint group_begin = treerange.first
                                       + intceil(static_cast<gpu_uint>(gpu_uint64(treerange.second - treerange.first)
                                                                       * group_id() / num_groups()),
                                                 gpu_uint(2));
                gpu_uint group_end = treerange.first
                                     + intceil(static_cast<gpu_uint>(gpu_uint64(treerange.second - treerange.first)
                                                                     * (group_id() + 1) / num_groups()),
                                               gpu_uint(2));

                group_end = min(group_end, treerange.second);

                gpu_for_local(group_begin, group_end, [&](gpu_uint self) {
                    // gpu_uint self = group_self + local_id();
                    gpu_bool has_children = scratch_tree[self].need_split;
                    {
                        for (uint si = 0; si < num_surrounding(); ++si)
                        {
                            gpu_uint cousin = surrounding_link(self, si);
                            has_children = has_children || (scratch_tree[cousin].need_split);
                        }
                    }
                    // If has_children is true, this node must further be split. Either because force
                    // calculation wants to continue in this node, or because it is used by another node
                    // in the vicinity.
                    scratch_tree[self].has_children = has_children;

                    Node_memory_request<gpu_bool> my = {
                        .split = scratch_tree[self].need_split,
                        .withchild = !scratch_tree[self].need_split && has_children && !scratch_tree[self].is_partnode,
                        .partnode_withchild =
                            !scratch_tree[self].need_split && has_children && scratch_tree[self].is_partnode,
                        .partnode_leaf =
                            !scratch_tree[self].need_split && !has_children && scratch_tree[self].is_partnode,
                        .leaf = !scratch_tree[self].need_split && !has_children && !scratch_tree[self].is_partnode
                    };

                    auto old = nmr;
                    nmr += my.cast<gpu_uint>();
                });

                nmr = work_group_reduce_add(nmr, local_size());

                gpu_if(local_id() == 0)
                {
                    // Writing memory requirements for this workgroup.
                    node_memory_request[group_id()] = nmr;
                }
            });

        this->node_memory_request.assign(device, treecount2.num_groups());

        treecount3.assign(
            device,
            [this](pair<gpu_uint, gpu_uint> treerange,
                   array<gpu_uint, 3> treeranges_parent_withchild_nochild,
                   gpu_T halflen,
                   gpu_uint8 depth,
                   Node_memory_request<gpu_uint> memory_offset) {
                auto old_children =
                    reinterpret<gpu_type<array<Tuint, 2>*>>(scratch.begin() + scratch_offset_old_children);
                auto old_node_p = scratch.begin() + scratch_offset_old_node;

                Node_memory_request<gpu_uint> next = memory_offset + this->node_memory_request[group_id()];
                auto oldnext = next;

                // Choosing loop order here in such a way that neighboring nodes in space will also be neighboring nodes
                // in memory. This loop must have the same ordering in treecount2 and treecount3.
                gpu_uint group_begin = treerange.first
                                       + intceil(static_cast<gpu_uint>(gpu_uint64(treerange.second - treerange.first)
                                                                       * group_id() / num_groups()),
                                                 gpu_uint(2));
                gpu_uint group_end = treerange.first
                                     + intceil(static_cast<gpu_uint>(gpu_uint64(treerange.second - treerange.first)
                                                                     * (group_id() + 1) / num_groups()),
                                               gpu_uint(2));

                group_end = min(group_end, treerange.second);

                const bool need_sync = (device.get_envmode() == env_CPU);

                gpu_for_local(
                    group_begin,
                    need_sync ? (group_begin + intceil(group_end - group_begin, (gpu_uint)local_size())) : group_end,
                    [&](gpu_uint old_self) {
                        // gpu_uint old_self = group_self + local_id();
                        // gpu_bool have_child = (tree_tmp[old_self].first_child != 0);

                        Node_memory_request<gpu_bool> my = zero;

                        gpu_if(old_self < group_end || !need_sync)
                        {
                            gpu_bool has_children = (gpu_bool)scratch_tree[old_self].has_children;
                            my = { .split = scratch_tree[old_self].need_split,
                                   .withchild = !scratch_tree[old_self].need_split && has_children
                                                && !scratch_tree[old_self].is_partnode,
                                   .partnode_withchild = !scratch_tree[old_self].need_split && has_children
                                                         && scratch_tree[old_self].is_partnode,
                                   .partnode_leaf = !scratch_tree[old_self].need_split && !has_children
                                                    && scratch_tree[old_self].is_partnode,
                                   .leaf = !scratch_tree[old_self].need_split && !has_children
                                           && !scratch_tree[old_self].is_partnode };
                        }

                        gpu_uint new_self;

                        new_self = cond(my.split, get_index(my.split, next.split, new_self), new_self);
                        new_self = cond(my.withchild, get_index(my.withchild, next.withchild, new_self), new_self);
                        new_self = cond(my.partnode_withchild,
                                        get_index(my.partnode_withchild, next.partnode_withchild, new_self),
                                        new_self);
                        new_self =
                            cond(my.partnode_leaf, get_index(my.partnode_leaf, next.partnode_leaf, new_self), new_self);
                        new_self = cond(my.leaf, get_index(my.leaf, next.leaf, new_self), new_self);

                        gpu_if(old_self < group_end || !need_sync)
                        {
                            /*
                          DUMP << "\ngroup_id=" << group_id() << ", local_id=" << local_id() << ": treecount3."
                             << "\nmy=" << my
                             << "\nnext: " << oldnext << " -> " << next
                             << "\nold_self=" << old_self << ", new_self=" << new_self
                             << "\ntree[" << new_self << "]=" << tree[new_self]
                             << "\nsetting tree[" << new_self << "].parent=" << parent_begin << " + (" << old_self <<
                          "-" <<  treerange.first << ")/2 = " << tree[new_self].parent << endl;
                            */

                            gpu_uint parent = treeranges_parent_withchild_nochild[0] + (old_self - treerange.first) / 2;
                            gpu_uint childnum = old_self % 2;
                            static_cast<scratch_treenode<gpu_T>&>(tree[new_self]) = scratch_tree[old_self];
                            tree[new_self].children = {};
                            tree[new_self].parent = parent;

                            // gpu_uint self = first_child + childnum;
                            Vector<gpu_T, 3> rcenter = tree[parent].rcenter;
                            rcenter[0] += cond(childnum == 0, -halflen, halflen);
                            tree[new_self].rcenter = rot(rcenter);
                            tree[new_self].signature = tree[parent].signature * 2 + childnum;

                            gpu_assert(parent >= 2u);
                            gpu_if(parent >= 2u)
                            {
                                get_children_p(tree.begin() + parent)[childnum] = new_self;
                                gpu_assert(old_node_p[new_self] == 0);
                                old_node_p[new_self] =
                                    reinterpret<gpu_type<Tuint*>>(old_children + old_node_p[parent])[childnum];
                            }
                        }
                    });

                gpu_for_global(treeranges_parent_withchild_nochild[1],
                               treeranges_parent_withchild_nochild[2],
                               [&](gpu_uint k) { tree[k].children = { 0, 0 }; });
            },
            this->treecount2.local_size(), // thread numbers must be the same as in treecount2.
            this->treecount2.global_size());

        treecount4_first_depth.assign(
            device, [this](pair<gpu_uint, gpu_uint> treerange, gpu_uint treebegin_next, gpu_T halflen_sublevel) {
                gpu_for_global(treerange.first, treerange.second, [&](gpu_uint parent) {
                    const gpu_uint first_child = treebegin_next + 2 * (parent - treerange.first);

                    auto old_p_range =
                        reinterpret<gpu_type<pair<Tuint, Tuint>*>>(scratch.begin() + scratch_offset_old_p_range);
                    auto old_children =
                        reinterpret<gpu_type<array<Tuint, 2>*>>(scratch.begin() + scratch_offset_old_children);
                    auto old_node_p = scratch.begin() + scratch_offset_old_node;

                    gpu_assert(old_children[old_node_p[parent]][0] != 0);
                    gpu_assert(tree[parent].pbegin == old_p_range[old_node_p[parent]].first);
                    gpu_assert(tree[parent].pend == old_p_range[old_node_p[parent]].second);
                    gpu_uint end = old_p_range[old_children[old_node_p[parent]][0]].second;

                    scratch_tree[first_child].pbegin = tree[parent].pbegin;
                    scratch_tree[first_child].pend = end;
                    scratch_tree[first_child + 1].pbegin = end;
                    scratch_tree[first_child + 1].pend = tree[parent].pend;

                    for (Tuint childnum : { 0, 1 })
                    {
                        gpu_uint self = first_child + childnum;
                        scratch_tree[self].depth = static_cast<gpu_uint8>(min_tree_depth);
                    }
                });
            });

        for (Tuint mod3 = 0; mod3 < 3; ++mod3)
        {
            treecount4[mod3].assign(
                device,
                [this, mod3](pair<gpu_uint, gpu_uint> treerange,
                             gpu_uint treebegin_next,
                             gpu_uint8 depth_sublevel,
                             gpu_T halflen_sublevel,
                             // resource<treenode<T>>& dest_tree,
                             resource<Vector<T, 3>>& x,
                             resource<Vector<T, 3>>& v,
                             resource<T>& mass) {
                    // gpu_ostream DUMP(cout);
                    gpu_for_global(treerange.first, treerange.second, [&](gpu_uint parent) {
                        const gpu_uint first_child = treebegin_next + 2 * (parent - treerange.first);

                        gpu_uint end;
                        {

                            auto old_p_range = reinterpret<gpu_type<pair<Tuint, Tuint>*>>(scratch.begin()
                                                                                          + scratch_offset_old_p_range);
                            auto old_children =
                                reinterpret<gpu_type<array<Tuint, 2>*>>(scratch.begin() + scratch_offset_old_children);
                            auto old_node_p = scratch.begin() + scratch_offset_old_node;

                            gpu_if(old_children[old_node_p[parent]][0] != 0)
                            {
                                gpu_assert(tree[parent].pbegin == old_p_range[old_node_p[parent]].first);
                                gpu_assert(tree[parent].pend == old_p_range[old_node_p[parent]].second);
                                end = old_p_range[old_children[old_node_p[parent]][0]].second;
                                // old_node_p[first_child] = old_children[old_node_p[parent]][0];
                                // old_node_p[first_child+1] = old_children[old_node_p[parent]][1];
                                // DUMP << "gid=" << global_id() << "setting old_node_p[" << first_child << "]=" <<
                                // old_node_p[first_child]
                                //<< " and old_node_p[" << first_child+1 << "]=" << old_node_p[first_child+1]
                                //<< endl;
                            }
                            gpu_else
                            {
                                // No data from previous tree. Need to sort particles to the two children
                                auto a = tree[parent].pbegin;
                                auto b = tree[parent].pend;

                                gpu_if(a != b)
                                {
                                    gpu_while(true)
                                    {
                                        gpu_while(a != b)
                                        {
                                            gpu_if(rot(x[a], mod3)[0] >= tree[parent].rcenter[0])
                                            {
                                                gpu_break();
                                            }
                                            ++a;
                                        }
                                        gpu_while(a != b)
                                        {
                                            gpu_if(rot(x[b - 1], mod3)[0] < tree[parent].rcenter[0])
                                            {
                                                gpu_break();
                                            }
                                            --b;
                                        }
                                        gpu_if(a == b)
                                        {
                                            gpu_break();
                                        }
                                        --b;
                                        gpu_assert(a < b);
                                        swap(v[a], v[b]);
                                        swap(x[a], x[b]);
                                        swap(mass[a], mass[b]);
                                        ++a;
                                    }
                                }
                                end = a;
                            }
                        }

                        scratch_tree[first_child].pbegin = tree[parent].pbegin;
                        scratch_tree[first_child].pend = end;
                        scratch_tree[first_child + 1].pbegin = end;
                        scratch_tree[first_child + 1].pend = tree[parent].pend;

                        for (Tuint childnum : { 0, 1 })
                        {
                            gpu_uint self = first_child + childnum;
                            scratch_tree[self].depth = depth_sublevel;
                        }
                    });
                });
        }

        make_tree_clear_leaf_childs.assign(device, [this](pair<gpu_uint, gpu_uint> treerange_leaf) {
            gpu_for_global(treerange_leaf.first, treerange_leaf.second, [&](gpu_uint k) {
                tree[k].children = { 0, 0 };
                static_cast<scratch_treenode<gpu_T>&>(tree[k]) = scratch_tree[k];
            });
        });

        {
            auto make_surrounding_func = [this](pair<gpu_uint, gpu_uint> treerange_parent,
                                                pair<gpu_uint, gpu_uint> treerange,
                                                gpu_T halflen,
                                                bool use_scratch,
                                                auto& dest_tree) {
                gpu_for_global(treerange_parent.first, treerange_parent.second, [&](gpu_uint parent) {
                    auto get_child = [&](gpu_uint node, Tuint child) {
                        gpu_assert(node < treerange.first);
                        if (use_scratch)
                        {
                            return cond(node - treerange_parent.first
                                            < treerange_parent.second - treerange_parent.first,
                                        treerange.first + 2 * (node - treerange_parent.first) + child,
                                        0u);
                        }
                        else
                        {
                            return tree[node].children[child];
                        }
                    };

                    for (Tuint c = 0; c < 2; ++c)
                    {
                        gpu_uint self = get_child(parent, c);
                        gpu_assert(self >= treerange.first);
                        gpu_assert(self < treerange.second);

                        for (Tuint si = 0; si < num_surrounding(); ++si)
                        {
                            Vector<Tint, 3> childvec = vdata.vicinity_vec[si];
                            Vector<Tint, 3> parentvec = vdata.child2parent(childvec, c);
                            auto parent_si_p = ranges::find(vdata.vicinity_vec, parentvec);

                            if (parent_si_p == vdata.vicinity_vec.end())
                            {
                                if (parentvec.squaredNorm() != 0)
                                {
                                    cout << "BAD: parentvec=" << parentvec
                                         << " not 0. Perhaps max_distfac is too small." << endl;
                                    throw std::runtime_error("BAD: parentvec not 0.");
                                }
                                // adding sibling
                                surrounding_link(self, si) = get_child(parent, 1 - c);
                            }
                            else
                            {
                                gpu_uint uncle = surrounding_link(parent, parent_si_p - vdata.vicinity_vec.begin());
                                surrounding_link(self, si) = get_child(uncle, (childvec[2] + c) % 2);
                            }
                            /*
                                            gpu_assert(
                                                surrounding_link(self, si) < 2u
                                                || (vdata.make_real(childvec, halflen * 2)
                                                    - (dest_tree[surrounding_link(self, si)].rcenter -
                               dest_tree[self].rcenter)) .squaredNorm() < 1E-4f * halflen);
                            */
                        }
                    }
                });
            };

            make_surrounding.assign(device,
                                    [this, make_surrounding_func](pair<gpu_uint, gpu_uint> treerange_parent,
                                                                  pair<gpu_uint, gpu_uint> treerange,
                                                                  gpu_T halflen) {
                                        make_surrounding_func(treerange_parent, treerange, halflen, false, this->tree);
                                    });
            make_surrounding_with_scratch.assign(
                device,
                [this, make_surrounding_func](
                    pair<gpu_uint, gpu_uint> treerange_parent, pair<gpu_uint, gpu_uint> treerange, gpu_T halflen) {
                    make_surrounding_func(treerange_parent, treerange, halflen, true, this->scratch_tree);
                });
        }

        movefunc.assign(device, [this](gpu_T dt, resource<Vector<T, 3>>& v, resource<Vector<T, 3>>& x) {
            gpu_for_global(0, x.size(), [&](gpu_uint k) {
                Vector<gpu_T, 3> old_x = x[k];

                x[k] += Vector<gpu_T, 3>(v[k]) * dt;

                gpu_if(x[k].cwiseAbs().maxCoeff() >= top_halflen)
                {
                    x[k] = old_x;
                    v[k] = { 0, 0, 0 };
                }
                gpu_assert(isfinite(x[k].squaredNorm()));
            });
        });

        kick.assign(device, [this](gpu_T dt, const resource<Vector<T, 3>>& force, resource<Vector<T, 3>>& v) {
            gpu_for_global(0, v.size(), [&](gpu_uint k) {
                v[k] += force[k] * dt;
                gpu_assert(isfinite(v[k].squaredNorm()));
            });
        });

        apply_vec2.assign(
            device, [this](const resource<Vector<T, 3>>& in, resource<Vector<T, 3>>& out) { apply_func(in, out); });
        apply_scalar2.assign(device, [this](const resource<T>& in, resource<T>& out) { apply_func(in, out); });

        for (unsigned int mod3 = 0; mod3 < 3; ++mod3)
        {

            upwards[mod3].assign(
                device,
                [mod3, this](array<gpu_uint, 3> treeranges_withchild_nochild,
                             gpu_T scale,
                             const resource<Vector<T, 3>>& x,
                             const resource<T>& mass) {
                    gpu_for_global(treeranges_withchild_nochild[0], treeranges_withchild_nochild[1], [&](gpu_uint t) {
                        gpu_matter_multipole Msum_r = zero;

                        for (int child = 0; child < 2; ++child)
                        {
                            const gpu_uint child_id = this->tree[t].children[child];
                            const gpu_matter_multipole Mcr = matter_tree[child_id];
                            Vector<double, 3> shift_r = { (1.0 - (2.0 * child)), 0, 0 };
                            gpu_matter_multipole Mr = Mcr.rot(-1).scale_ext(pow<-1, 3>(2.0)).shift_ext(shift_r);
                            Msum_r += Mr;
                        }
                        matter_tree[t] = Msum_r;
                    });

                    gpu_for_global(treeranges_withchild_nochild[1], treeranges_withchild_nochild[2], [&](gpu_uint t) {
                        gpu_matter_multipole Msum_r = zero;

                        gpu_for(this->tree[t].pbegin, this->tree[t].pend, [&](gpu_uint p) {
                            allow_preload();
                            Msum_r += gpu_matter_multipole::from_particle(
                                ((rot(x[p], mod3) - this->tree[t].rcenter) * scale).eval(), mass[p]);
                        });
                        matter_tree[t] = Msum_r;
                    });
                });
        }

        for (bool with_potential : { false, true })
        {
            for (unsigned int mod3 = 0; mod3 < 3; ++mod3)
            {
                handle_particles_from_multipole[with_potential][mod3].assign(
                    device,
                    [with_potential, mod3, this](pair<gpu_uint, gpu_uint> partnode_range,
                                                 gpu_T scale,
                                                 const resource<Vector<T, 3>>& x,
                                                 resource<Vector<T, 3>>& force,
                                                 resource<T>& potential) {
                        gpu_for_group(partnode_range.first, partnode_range.second, [&](gpu_uint self) {
                            gpu_assert(this->tree[self].is_partnode);
                            gpu_assert(!this->tree[self].need_split && this->tree[this->tree[self].parent].need_split);

                            gpu_for_local(this->tree[self].pbegin, this->tree[self].pend, [&](gpu_uint pa) {
                                allow_preload();
                                gpu_T halflen = pow<1, 3>(2.f) / scale;
                                Vector<gpu_T, 3> halflen3 = vdata.make_real({ 1, 1, 1 }, halflen);
                                for (uint k = 0; k < 3; ++k)
                                {
                                    gpu_assert(abs((rot(x[pa], mod3) - this->tree[self].rcenter)[k])
                                               < halflen3[k] * 1.001f);
                                }

                                Vector<gpu_T, 3> F =
                                    rot(force_tree[self].calc_force(
                                            ((rot(x[pa], mod3) - this->tree[self].rcenter) * scale).eval()),
                                        -(Tint)mod3)
                                    * pow2(scale);

#if CALC_POTENTIAL
                                gpu_T P = force_tree[self].calc_loc_potential(
                                              ((rot(x[pa], mod3) - this->tree[self].rcenter) * scale).eval())
                                          * scale;
#endif

                                force[pa] += F;
                                gpu_assert(isfinite(force[pa].squaredNorm()));
#if CALC_POTENTIAL
                                if (with_potential)
                                {
                                    potential[pa] += P;
                                }
#endif
                            });
                        });
                    });
            }

            handle_particles_direct2[with_potential].assign(
                device,
                [this, with_potential](const resource<Vector<T, 3>>& x,
                                       const resource<T>& mass,
                                       resource<Vector<T, 3>>& force,
                                       resource<T>& potential) {
                    vector<Tuint> cats = { 16, 8, 4, 3, 2, 1 };

                    vector<local_mem<pair<Tuint, Tuint>>> todo;
                    for (Tuint c : cats)
                    {
                        todo.emplace_back(local_size() * 2);
                    }
                    vector<gpu_uint> num_todo(cats.size(), 0);

                    auto doit = [&](Tuint c) {
                        auto [self, pbegin] = todo[c][num_todo[c] + local_id()];
                        vector<Vector<gpu_T, 3>> F(cats[c], { 0, 0, 0 });
                        vector<gpu_T> P(cats[c], 0);
                        gpu_uint N = (tree[self].pend - pbegin);
                        if (c == 0)
                        {
                            N = min(N, cats[c]);
                        }
                        Tuint min_N = 1;
                        if (c + 1 < cats.size())
                        {
                            min_N = cats[c + 1] + 1;
                            assert(cats[c] <= 2 * cats[c + 1]);
                        }
                        if (cats[c] == min_N)
                        {
                            gpu_assert(N == cats[c]);
                            N = cats[c];
                        }

                        gpu_assert(N <= cats[c]);
                        vector<Vector<gpu_T, 3>> my_x(cats[c]);
                        for (Tuint k = 0; k < cats[c]; ++k)
                        {
                            gpu_if(k < N || k < min_N)
                            {
                                my_x[k] = x[pbegin + k];
                            }
                        }

                        // Starting with the self node, since it is not included in the surrounding list.
                        gpu_int si = -1;
                        gpu_uint other_node = self; // surrounding_link(self, si);
                        gpu_uint other_p = tree[other_node].pbegin;
                        gpu_uint other_pend = tree[other_node].pend;

                        gpu_bool docontinue = true;
                        gpu_while(docontinue)
                        {
                            for (Tuint k = 0; k < cats[c]; ++k)
                            {
                                const Vector<gpu_T, 3> dist = x[other_p] - my_x[k];
                                F[k] += dist
                                        * (mass[other_p] * pow<-1, 2>(dist.squaredNorm() + 1E-20f)
                                           * pow2(pow<-1, 2>(dist.squaredNorm() + 1E-20f)));

                                P[k] += cond(dist.squaredNorm() == 0,
                                             0.f,
                                             -(mass[other_p] * pow<-1, 2>(dist.squaredNorm() + 1E-20f)));
                            }
                            ++other_p;
                            gpu_while(other_p == other_pend)
                            {
                                ++si;
                                gpu_if(si == num_surrounding())
                                {
                                    docontinue = false;
                                    gpu_break();
                                }
                                other_node = surrounding_link(self, si);
                                other_p = tree[other_node].pbegin;
                                other_pend = tree[other_node].pend;
                            }
                        }
                        for (Tuint k = 0; k < cats[c]; ++k)
                        {
                            gpu_if(k < N || k < min_N)
                            {
                                force[pbegin + k] = F[k];
                                if (with_potential)
                                {
                                    potential[pbegin + k] = P[k];
                                }
                            }
                        }
                    };

                    auto node_p = resource(partnode_ranges_packed_buf).cbegin();
                    gpu_uint self = global_id() + node_p->first;

                    gpu_while(node_p->second != 0)
                    {
                        gpu_uint group_end =
                            node_p->first + intceil(node_p->second - node_p->first, (gpu_uint)local_size());
                        gpu_while(self < group_end)
                        {
                            gpu_uint pbegin = 0;
                            gpu_uint pend = 0;
                            gpu_if(self < node_p->second)
                            {
                                pbegin = tree[self].pbegin;
                                pend = tree[self].pend;
                            }

                            for (Tuint c = 0; c < cats.size(); ++c)
                            {
                                auto run_c = [&]() {
                                    gpu_bool yes;
                                    if (c + 1 < cats.size())
                                    {
                                        yes = gpu_int(pend - pbegin) > (int)cats[c + 1];
                                    }
                                    else
                                    {
                                        yes = gpu_int(pend - pbegin) > 0;
                                    }
                                    gpu_uint i = get_index(yes, num_todo[c]);

                                    gpu_if(yes)
                                    {
                                        todo[c][i] = { self, pbegin };
                                        pbegin += cats[c];
                                    }

                                    gpu_if(num_todo[c] >= local_size())
                                    {
                                        num_todo[c] -= local_size();
                                        local_barrier(memory::threadgroup);
                                        doit(c);
                                        local_barrier(memory::threadgroup);
                                    }
                                };

                                if (c == 0)
                                {
                                    gpu_while(work_group_any(gpu_int(pend - pbegin) > (int)cats[0] + (int)cats[1]))
                                    {
                                        run_c();
                                    }
                                }
                                run_c();
                            }
                            self += global_size();
                        }
                        self += node_p[1].first - group_end;
                        ++node_p;
                    }

                    local_barrier(memory::threadgroup);
                    for (Tuint c = 0; c < cats.size(); ++c)
                    {
                        gpu_if(local_id() < num_todo[c])
                        {
                            num_todo[c] = 0;
                            doit(c);
                        }
                    }
                });

            downwards2[with_potential].assign(device, [this, with_potential](pair<gpu_uint, gpu_uint> split_range) {
                gpu_for_global(split_range.first, split_range.second, [&](gpu_uint parent) {
                    gpu_assert(this->tree[parent].need_split);

                    array<gpu_force_multipole, 2> new_multipole;
                    for (int c = 0; c < 2; ++c)
                    {
                        Vector<Tdouble, 3> shiftvec = { 0, 0, pow<1, 3>(2.0) * (2 * c - 1) };
                        new_multipole[c] = force_tree[parent].rot().scale_loc(pow<1, 3>(2.0)).shift_loc(shiftvec);
                    }

                    Tuint count = 0;
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

                        for (int c = 0; c < 2; ++c)
                        {
                            for (int c2 = 0; c2 < 2; ++c2)
                            {
                                Vector<Tint, 3> cousinvec = vdata.parent2child(unclevec, c2 - c);
                                if (!vdata.local_indices.contains(vdata.make_index(cousinvec))
                                    && cousinvec.squaredNorm() != 0)
                                {
                                    gpu_uint cousin = this->tree[(parent_si == this->num_surrounding()
                                                                      ? parent
                                                                      : this->surrounding_link(parent, parent_si))]
                                                          .children[c2];
                                    new_multipole[c] +=
                                        matter_tree[cousin].makelocal(vdata.make_real(-cousinvec, 2 * pow<1, 3>(2.0)));

                                    if (!with_potential)
                                    {
                                        // Marking the first order multipole as unused. This should speed up the
                                        // calculations if the potential is not required.
                                        new_multipole[c].template get<0>().A[0] = {};
                                    }
                                    ++count;
                                }
                            }
                        }
                    }

                    assert(count == (this->num_surrounding() + 1) * 2);

                    for (int c = 0; c < 2; ++c)
                    {
                        gpu_uint self = this->tree[parent].children[c];
                        force_tree[self] = new_multipole[c];
                    }
                });
            });
            cout << "created downwards." << endl;
        }

        update_tree_set_nodelink.assign(device, [this]() {
            auto nodelink = scratch.begin() + scratch_offset_nodelink;

            multi_level_loop(all_leaf_ranges_packed_buf, global_id(), global_size(), [&](gpu_uint self) {
                gpu_for(this->tree[self].pbegin, this->tree[self].pend, [&](gpu_uint p) { nodelink[p] = self; });
            });
        });

        update_tree_1.assign(device, [this](const resource<Vector<T, 3>>& x) {
            auto next_p = scratch.begin() + scratch_offset_next_p;
            // auto num_removed = scratch.begin() + scratch_offset_num_removed;
            auto particle_list = scratch.begin() + scratch_offset_particle_list;
            auto nodelink = scratch.begin() + scratch_offset_nodelink;

            gpu_for_global(0, x.size(), [&](gpu_uint p) {
                gpu_uint node = nodelink[p];
                gpu_uint depth = this->tree[node].depth;
                gpu_signature_t new_sig = calc_sig<signature_t>(x[p], MAX_DEPTH());
                // plist[p].first = new_sig;

                gpu_signature_t shifted_sig = new_sig >> ((Tuint)sizeof(signature_t) * 8 - depth);
                gpu_signature_t sig_change = shifted_sig ^ tree[node].signature;

                gpu_if(sig_change != 0)
                {
                    gpu_signature_t bit = (gpu_signature_t(1) << (2 * Tuint(sizeof(signature_t) * 8) - 1
                                                                  - countl_zero(sig_change) - depth));

                    sig_change /= 2;
                    gpu_uint i = this->tree[node].parent;

                    // Going up to the last node that contains both the old and the new signature.
                    gpu_while(sig_change != 0)
                    {
                        i = this->tree[i].parent;
                        sig_change /= 2;
                    }

                    gpu_uint child = get_children_p(this->tree.begin() + i)[(new_sig & bit) != 0];
                    // Going down to the new leaf.
                    gpu_while(child != 0)
                    {
                        i = child;
                        bit >>= 1;
                        child = get_children_p(this->tree.begin() + i)[(new_sig & bit) != 0];
                    }

                    // Add to linked list of particles that are to be added to node i.
                    particle_list[p] = atomic_xchg(next_p[i], p, std::memory_order_relaxed);
                }
            });
        });

        update_tree_upwards.assign(
            device, [this](pair<gpu_uint, gpu_uint> leaf_range, pair<gpu_uint, gpu_uint> withchild_range) {
                // gpu_ostream DUMP(cout);
                // auto num_added = scratch.begin();
                auto next_p = scratch.begin() + scratch_offset_next_p;
                // auto num_removed = scratch.begin() + scratch_offset_num_removed;
                auto particle_list = scratch.begin() + scratch_offset_particle_list;
                auto new_num_p = scratch.begin() + scratch_offset_new_num_p;

                gpu_for_global(leaf_range.first, leaf_range.second, [&](gpu_uint self) {
                    gpu_uint num_p = 0;

                    // First count how many of the previous particles are still in this node.
                    gpu_for(tree[self].pbegin, tree[self].pend, [&](gpu_uint p) {
                        num_p += (particle_list[p] == 0xfffffffe);
                    });

                    // Now count new particles from other nodes.
                    gpu_uint p = next_p[self];
                    gpu_while(p != ~0u)
                    {
                        p = particle_list[p];
                        ++num_p;
                    }
                    new_num_p[self] = num_p;
                });

                gpu_for_global(withchild_range.first, withchild_range.second, [&](gpu_uint self) {
                    gpu_uint num_p = 0;
                    for (Tuint c = 0; c < 2; ++c)
                    {
                        gpu_uint child = this->tree[self].children[c];
                        num_p += new_num_p[child];
                    }
                    new_num_p[self] = num_p;

                    // DUMP << "upwards_split. depth=" << this->tree[self].depth << ", new_num_p[" << self << "]=" <<
                    // new_num_p[self] << "\n";
                });
            });

        update_tree_downwards.assign(device, [this](pair<gpu_uint, gpu_uint> treerange_parent) {
            // gpu_ostream DUMP(cout);
            auto new_num_p = scratch.begin() + scratch_offset_new_num_p;
            auto old_p_range = reinterpret<gpu_type<pair<Tuint, Tuint>*>>(scratch.begin() + scratch_offset_old_p_range);

            gpu_for_global(treerange_parent.first, treerange_parent.second, [&](gpu_uint parent) {
                gpu_uint pbegin = this->tree[parent].pbegin;
                for (uint child = 0; child < 2; ++child)
                {
                    gpu_uint self = tree[parent].children[child];
                    old_p_range[self] = { tree[self].pbegin, tree[self].pend };
                    this->tree[self].pbegin = pbegin;
                    pbegin += new_num_p[self];
                    this->tree[self].pend = pbegin;
                    /*
                      DUMP << "downwards. depth=" << this->tree[self].depth
                     << ", self=" << self
                     << ", new_num_p=" << new_num_p[self]
                     << ", pbegin=" << this->tree[self].pbegin << ", pend=" << this->tree[self].pend << "\n";
                    */
                }
            });
        });

        const Tuint scratch_end = scratch_offset_old_node + tree.size();

        cout << "scratch usage: " << scratch_end << " / " << scratch.size() << endl;
        if (scratch_end > scratch.size())
        {
            cout << "scratch size to small." << endl;
            exit(1);
        }

        update_tree_reorder_particles.assign(device, [this]() {
            auto particle_list = scratch.begin() + scratch_offset_particle_list;
            auto next_p = scratch.begin() + scratch_offset_next_p;
            auto old_p_range = reinterpret<gpu_type<pair<Tuint, Tuint>*>>(scratch.begin() + scratch_offset_old_p_range);
            auto new_order = scratch.begin() + scratch_offset_new_order;

            multi_level_loop(all_leaf_ranges_packed_buf, global_id(), global_size(), [&](gpu_uint self) {
                gpu_uint new_pbegin = this->tree[self].pbegin;
                gpu_uint new_pend = this->tree[self].pend;

                gpu_uint dest = new_pbegin;

                gpu_for(old_p_range[self].first, old_p_range[self].second, [&](gpu_uint p) {
                    gpu_if(particle_list[p] == 0xfffffffe)
                    {
                        new_order[dest] = p;
                        ++dest;
                    }
                });
                gpu_uint p = next_p[self];
                gpu_while(p != ~0u)
                {
                    new_order[dest] = p;
                    ++dest;
                    p = particle_list[p];
                }
            });
        });
    }
};
