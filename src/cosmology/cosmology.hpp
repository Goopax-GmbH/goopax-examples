#define WITH_TIMINGS 0

#if WITH_OPENCV
#include <opencv2/opencv.hpp>
#endif

#include <SDL3/SDL_main.h>
#include <goopax_draw/particle.hpp>
#if WITH_TIMINGS
#include <chrono>
#endif
#include "../common/output.hpp"
#include <fstream>
#include <goopax_draw/window_sdl.h>
#include <goopax_extra/param.hpp>
#include <goopax_extra/struct_types.hpp>
#include <random>
#include <set>

#define SURROUNDING1 0
#define SURROUNDING2 1

#define MULTIPOLE_ORDER 4
#define HAVE_BFLOAT16 1

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

using signature_t = Tuint64_t;
using gpu_signature_t = typename make_gpu<signature_t>::type;

#define CALC_POTENTIAL 1

#include "multipole_cart.hpp"

const float top_halflen = 4;
PARAMOPT<Tuint> MAX_DEPTH("max_depth", 64);

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
    }
    static Vector<Tint, 3> child2parent(Vector<Tint, 3> p, Tint c)
    {
        p = rot(p, -1);
        p[0] = floor((p[0] + c) / 2.0);
        return p;
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

template<typename T>
struct CosmosData
{
    buffer<Vector<T, 3>> x;
    buffer<Vector<T, 3>> v;
#if CALC_POTENTIAL
    buffer<T> potential;
#endif
    buffer<T> mass;
    buffer<Vector<T, 3>> force;
    buffer<T> tmps;

    void swapBuffers(bool preserve_potential)
    {
        // mimicking the buffer swap in update_tree

        swap(x, force);
        swap(v, force);
        swap(mass, tmps);
        if (preserve_potential)
        {
            swap(potential, tmps);
        }
    }
};

template<class T, unsigned int max_multipole>
struct Cosmos : public CosmosData<T>
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
    using gpu_force_multipole_f32 = multipole<gpu_float, gpu_float, gpu_float, gpu_float>;
    using gpu_force_multipole_bf16 = multipole<gpu_bfloat16, gpu_bfloat16, gpu_bfloat16, gpu_bfloat16>;
#elif MULTIPOLE_ORDER == 4
#if HAVE_BFLOAT16
    using matter_multipole = multipole<Tbfloat16, Tbfloat16, T, T, T>;
    using force_multipole = multipole<Tbfloat16, Tbfloat16, T, T, T>;
    using gpu_force_multipole_f32 = multipole<gpu_float, gpu_float, gpu_float, gpu_float, gpu_float>;
    using gpu_force_multipole_bf16 = multipole<gpu_bfloat16, gpu_bfloat16, gpu_bfloat16, gpu_bfloat16, gpu_bfloat16>;
#else
    using matter_multipole = multipole<T, T, T, T, T>;
    using force_multipole = multipole<T, T, T, T, T>;
    using gpu_force_multipole_f32 = multipole<gpu_float, gpu_float, gpu_float, gpu_float, gpu_float>;
    using gpu_force_multipole_bf16 = multipole<gpu_float, gpu_float, gpu_float, gpu_float, gpu_float>;
#endif
#else
#error
#endif
    using gpu_matter_multipole = typename make_gpu<matter_multipole>::type;
    using gpu_force_multipole = typename make_gpu<force_multipole>::type;

    goopax_device device;
    const Tuint num_particles;
    using gpu_T = typename make_gpu<T>::type;
    const vicinity_data vdata;

    const Tuint max_treesize;

    const unsigned int max_nodesize;

    buffer<treenode<T>> tree;

#if SURROUNDING1
    buffer<Tuint> surrounding_buf;
#endif
#if SURROUNDING2
    buffer<array<Tuint, 6>> surrounding2_buf;
#endif

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

    unsigned int num_surrounding() const
    {
        return vdata.vicinity_vec.size();
    }

#if SURROUNDING1
    gpu_uint& surrounding_link(gpu_uint node, gpu_uint k)
    {
        gpu_assert(k < num_surrounding());
        return surrounding_buf[node * num_surrounding() + k];
    }
#endif

#if SURROUNDING2
    vector<tuple<gpu_uint, Tuint, Vector<Tint, 3>>>
    get_surrounding(gpu_uint self, gpu_uint parent, Tuint max_depth, bool with_self, bool use_scratch) const
    {
        assert(max_depth <= 3);

        assert(max_depth < min_tree_depth);
        vector<tuple<gpu_uint, Tuint, Vector<Tint, 3>>> ret;

        set<Tuint> have;
        if (!with_self)
        {
            have.insert(vdata.make_index({ 0, 0, 0 }));
        }

        array<gpu_uint, 3> reverse = { 0, 0, 0 };
        if (use_scratch)
        {
            for (Tuint depth = 0; depth < max_depth; ++depth)
            {
                if (depth == 0)
                {
                    reverse[0] = self % 2;
                }
                else
                {
                    reverse[depth] = (static_cast<gpu_uint>(tree[parent].signature) & (1 << (depth - 1))) != 0;
                }
            }
        }
        else
        {
            for (Tuint depth = 0; depth < max_depth; ++depth)
            {
                reverse[depth] = (static_cast<gpu_uint>(tree[self].signature) & (1 << depth)) != 0;
            }
        }

        for (int depth = max_depth; depth >= 0; --depth)
        {
            auto get_neighbor = [&](gpu_uint node, int dim, bool dir) {
                return gpu_array_access(surrounding2_buf[node], 2 * dim + (dir ^ (reverse[(1 + dim + depth) % 3])));
            };

            auto get_all_points = [depth](Vector<Tint, 3> p) {
                vector<Vector<Tint, 3>> ret = { p };

                for (int d = 0; d < depth; ++d)
                {
                    vector<Vector<Tint, 3>> tmp;
                    for (auto p : ret)
                    {
                        p[0] *= 2;
                        tmp.push_back(rot(p));
                        p[0] += 1;
                        tmp.push_back(rot(p));
                    }
                    swap(tmp, ret);
                }
                return ret;
            };

            gpu_uint node = self;
            for (int k = 0; k < depth; ++k)
            {
                if (k == 0)
                {
                    node = parent;
                }
                else
                {
                    node = tree[node].parent;
                }
            }

            Vector<Tint, 3> p;
            for (int dz : { -1, 1 })
            {
                gpu_uint node_z = node;
                for (p[2] = 0;; p[2] += dz, node_z = get_neighbor(node_z, 2, (dz == 1)))
                {
                    for (int dy : { -1, 1 })
                    {
                        gpu_uint node_y = node_z;
                        for (p[1] = 0;; p[1] += dy, node_y = get_neighbor(node_y, 1, (dy == 1)))
                        {
                            for (int dx : { -1, 1 })
                            {
                                gpu_uint node_x = node_y;
                                for (p[0] = 0;; p[0] += dx, node_x = get_neighbor(node_x, 0, (dx == 1)))
                                {
                                    vector<Vector<Tint, 3>> points = get_all_points(p);

                                    bool all_in_range = true;
                                    bool some_in_range = false;
                                    bool have_some = false;
                                    for (auto p : points)
                                    {
                                        all_in_range =
                                            all_in_range && vdata.local_indices.contains(vdata.make_index(p));
                                        some_in_range =
                                            some_in_range || vdata.local_indices.contains(vdata.make_index(p));
                                        have_some = have_some || have.contains(vdata.make_index(p));
                                    }
                                    if (!some_in_range)
                                    {
                                        break;
                                    }
                                    if (all_in_range && !have_some)
                                    {
                                        ret.push_back({ node_x, depth, p });
                                        for (auto p : points)
                                        {
                                            have.insert(vdata.make_index(p));
                                        }
                                    }
                                }
                            }
                            if (p[0] == 0)
                                break;
                        }
                    }
                    if (p[1] == 0)
                        break;
                }
            }
        }

        assert(have.size() == num_surrounding() + 1);
        return ret;
    }
#endif

    buffer<Node_memory_request<>> node_memory_request;

    void print(auto& tree, pair<Tuint, Tuint> treerange, bool with_surrounding = false)
    {
        const_buffer_map map(tree);
#if SURROUNDING1
        const_buffer_map surrounding(this->surrounding_buf);
#endif
#if SURROUNDING2
        const_buffer_map surrounding2(this->surrounding2_buf);
#endif
        for (uint k = treerange.first; k < treerange.second; ++k)
        {
            cout << k << ": " << map[k];
            if (with_surrounding)
            {
#if SURROUNDING1
                cout << ". surrounding:";
                for (uint s = 0; s < this->num_surrounding(); ++s)
                {
                    cout << " " << surrounding[k * this->num_surrounding() + s];
                }
#endif
#if SURROUNDING2
                cout << ". surrounding2:";
                cout << " " << surrounding2[k];
#endif
            }
            cout << endl;
        }
    }

    kernel<void(pair<Tuint, Tuint> treerange)> save_old_tree_data;

    kernel<void(pair<Tuint, Tuint> treerange, Tuint parent_begin)> treecount1;

    kernel<void(pair<Tuint, Tuint> treerange, Tuint parent_begin, buffer<Node_memory_request<>>& node_memory_request)>
        treecount2;

    kernel<void(pair<Tuint, Tuint> treerange,
                array<Tuint, 3> treeranges_parent_withchild_nochild,
                T halflen,
                Node_memory_request<> memory_offset)>
        treecount3;

    kernel<void(pair<Tuint, Tuint> treerange_with_child, Tuint treebegin_next)> treecount4_first_depth;

    array<kernel<void(pair<Tuint, Tuint> treerange_with_child,
                      Tuint treebegin_next,
                      Tuint8_t depth_sublevel,
                      buffer<Vector<T, 3>>& x,
                      buffer<Vector<T, 3>>& v,
                      buffer<T>& mass)>,
          3>
        treecount4;

    kernel<void(pair<Tuint, Tuint> treerange_leaf)> make_tree_clear_leaf_childs;

    kernel<void(pair<Tuint, Tuint> treerange_parent, pair<Tuint, Tuint> treerange)> make_surrounding;
    kernel<void(pair<Tuint, Tuint> treerange_parent, pair<Tuint, Tuint> treerange)> make_surrounding_with_scratch;

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

    void make_tree_base()
    {
        cout << "make_tree_base" << endl;
#if SURROUNDING1
        surrounding_buf.fill(0, 0, 4 * num_surrounding());
#endif
#if SURROUNDING2
        surrounding2_buf.fill(zero, 0, 4);
#endif
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
            make_surrounding(treeranges[depth - 1], treeranges[depth]).wait();
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
                            gpu_size_t(self - partnode_leaf_ranges.back().first) * this->x.size()
                            / (partnode_leaf_ranges.back().second - partnode_leaf_ranges.back().first));
                        tree[self].pend = static_cast<gpu_uint>(
                            gpu_size_t(self - partnode_leaf_ranges.back().first + 1) * this->x.size()
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
#if SURROUNDING1
        surrounding_buf.fill({}, treeoffset * num_surrounding(), surrounding_buf.size());
#endif
#if SURROUNDING2
        surrounding2_buf.fill({}, treeoffset, surrounding2_buf.size());
#endif
#endif

        scratch_tree.fill(zero, 0, 2);

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
                treeranges[min_tree_depth - 1].second);

            for (Tuint depth = min_tree_depth; depth < MAX_DEPTH(); ++depth)
            {
                pair<Tuint, Tuint> treerange = { treeoffset, treeoffset + treesize };
                // cout << "depth=" << depth << ", treerange=" << treerange << endl;

                if (treerange.second - treeranges.back().first > force_tree.size())
                {
                    cout << "force tree too small for treeranges " << treeranges.back() << " and " << treerange << endl;
                    throw std::runtime_error("force_tree is too small to contain two adjacent tree depth ranges");
                }

                this->treeranges.push_back(treerange);

                if (depth == MAX_DEPTH() - 1)
                    break;

#if WITH_TIMINGS
                device.wait_all();
                auto g0 = steady_clock::now();
#endif

                make_surrounding_with_scratch(
                    { treeranges_withchild_nochild[depth - 1][0], treeranges_withchild_nochild[depth - 1][1] },
                    treerange)
                    .wait();

#if WITH_TIMINGS
                device.wait_all();
                auto g1 = steady_clock::now();
#endif
                // cout << "calling treecount1. treerange=" << treerange << endl;
                this->treecount1(treerange, treeranges[depth - 1].first).wait();

#if WITH_TIMINGS
                device.wait_all();
                auto g2 = steady_clock::now();
#endif

                this->treecount2(treerange, treeranges[depth - 1].first, this->node_memory_request);

#if WITH_TIMINGS
                device.wait_all();
                auto g4 = steady_clock::now();
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
                }

                assert(treeoffset == treerange.second);

                if (treerange.second + treesize > tree.size())
                {
                    throw std::runtime_error("tree too small");
                }

                treecount3(treerange,
                           (depth == 0 ? array<Tuint, 3>{ 0, 0, 0 } : treeranges_withchild_nochild[depth - 1]),
                           level_halflen,
                           memory_offset);

                // cout << "after treecount3: tree:\n";
                // print(this->tree, treerange);

#if WITH_TIMINGS
                device.wait_all();
                auto g5 = steady_clock::now();
#endif

                make_surrounding(
                    { treeranges_withchild_nochild[depth - 1][0], treeranges_withchild_nochild[depth - 1][1] },
                    treerange)
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
                                      this->x,
                                      this->v,
                                      this->mass);
#if WITH_TIMINGS
                device.wait_all();
                auto g7 = steady_clock::now();
#endif

                Vector<T, 3> boxsize;
                for (Tuint k = 0; k < 3; ++k)
                {
                    boxsize[k] = top_halflen * (pow(2.0, -Tint(depth + 2 - k) / 3 + (2.0 - k) / 3.0) + 1E-7);
                }

                if (treesize == 0)
                    break;

                level_halflen *= pow(2.0, -1.0 / 3);

#if WITH_TIMINGS
                treetime[0] += (g1 - g0);
                treetime[1] += (g2 - g1);
                // treetime[2] += (g3 - g2);
                treetime[3] += (g4 - g2);
                treetime[4] += (g5 - g4);
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
             << "    make_surrounding[1]: " << duration_cast<chrono::milliseconds>(treetime[0]) << endl
             << "    treecount1: " << duration_cast<chrono::milliseconds>(treetime[1]) << endl
             << "    treecount2: " << duration_cast<chrono::milliseconds>(treetime[3]) << endl
             << "    treecount3: " << duration_cast<chrono::milliseconds>(treetime[4]) << endl
             << "    make_surrounding[2]: " << duration_cast<chrono::milliseconds>(treetime[5]) << endl
             << "    treecount4: " << duration_cast<chrono::milliseconds>(treetime[6]) << endl;
#endif
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
        const Tuint np = min(this->x.size(), (Tuint)100);

        goopax_future<double> poterr;
        Tdouble tot = verify(this->x,
                             this->force,
                             this->mass,
                             this->potential,
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

    static constexpr unsigned int scratch_offset = 4 * sizeof(matter_multipole) / sizeof(Tuint);

    const Tuint scratch_offset_tree = scratch_offset;

    const Tuint scratch_offset_next_p = scratch_offset;
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
                upwards[depth % 3](this->treeranges_withchild_nochild[depth], scale, this->x, this->mass);

                scale *= pow<-1, 3>(2.0);
            }
        }

#if WITH_TIMINGS
        device.wait_all();
        auto t2 = steady_clock::now();
#endif

        this->handle_particles_direct2[with_potential](this->x, this->mass, this->force, this->potential);

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

                handle_particles_from_multipole[with_potential][depth % 3](
                    this->partnode_ranges[depth], scale, this->x, this->force, this->potential);
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
        scratch.fill(~0u, scratch_offset_next_p, scratch_offset_next_p + this->treeranges.back().second);
        scratch.fill(0xfffffffe, scratch_offset_particle_list, scratch_offset_particle_list + this->x.size());
        scratch.fill(0, scratch_offset_new_num_p, scratch_offset_new_num_p + 2);

#if WITH_TIMINGS
        device.wait_all();
        auto t0 = steady_clock::now();
#endif

        update_tree_set_nodelink();

#if WITH_TIMINGS
        device.wait_all();
        auto t1 = steady_clock::now();
#endif
        update_tree_1(this->x);

#if WITH_TIMINGS
        device.wait_all();
        auto t2 = steady_clock::now();
#endif

        for (Tint depth = this->treeranges.size() - 1; depth != -1; --depth)
        {
            update_tree_upwards({ this->partnode_leaf_ranges[depth].first, treeranges[depth].second },
                                { this->treeranges[depth].first, this->partnode_withchild_ranges[depth].second });
        }

#if WITH_TIMINGS
        device.wait_all();
        auto t3 = steady_clock::now();
#endif

        for (Tuint depth = 1; depth < this->treeranges.size(); ++depth)
        {
            update_tree_downwards(
                { treeranges_withchild_nochild[depth - 1][0], treeranges_withchild_nochild[depth - 1][1] });
        }
#if WITH_TIMINGS
        device.wait_all();
        auto t4 = steady_clock::now();
#endif

        update_tree_reorder_particles();

#if WITH_TIMINGS
        device.wait_all();
        auto t5 = steady_clock::now();
#endif

        this->apply_vec2(this->x, this->force);
        swap(this->x, this->force);
        this->apply_vec2(this->v, this->force);
        swap(this->v, this->force);
        this->apply_scalar2(this->mass, this->tmps);
        swap(this->mass, this->tmps);
        if (preserve_potential)
        {
            this->apply_scalar2(this->potential, this->tmps);
            swap(this->potential, this->tmps);
        }

#if WITH_TIMINGS
        device.wait_all();
        auto t6 = steady_clock::now();
#endif

#if GOOPAX_DEBUG
        scratch.fill({}, scratch_offset, scratch.size());
#endif

#if WITH_TIMINGS
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

    Cosmos(CosmosData<T>&& data,
           size_t max_treesize0,
           size_t min_force_tree_size,
           Tdouble max_distfac,
           unsigned int max_nodesize0)

        : CosmosData<T>(std::move(data))
        , device(this->x.get_device())
        , num_particles(this->x.size())
        , vdata(max_distfac)
        , max_treesize(max_treesize0)
        , max_nodesize(max_nodesize0)
        , tree(device, max_treesize)
#if SURROUNDING1
        , surrounding_buf(device, max_treesize * num_surrounding())
#endif
#if SURROUNDING2
        , surrounding2_buf(device, max_treesize)
#endif
        , partnode_ranges_packed_buf(device, MAX_DEPTH() + 1)
        , all_leaf_ranges_packed_buf(device, MAX_DEPTH() + 1)
        , matter_tree(device, this->max_treesize)
        , scratch(reinterpret<buffer<Tuint>>(matter_tree))
    {
        std::stack<future<void>> futures;
        // constexpr std::launch policy = std::launch::deferred;
        constexpr std::launch policy = std::launch::async;

        {
            size_t force_tree_size = 1;
            while (force_tree_size < min_force_tree_size
                   || force_tree_size * sizeof(force_multipole) < tree.size() * sizeof(scratch_treenode<T>))
            {
                force_tree_size *= 2;
            }
            cout << "using force_tree_size=" << force_tree_size << endl;
            force_tree.assign(device, force_tree_size);
        }
        scratch_tree = (reinterpret<buffer<scratch_treenode<T>>>(force_tree));
        if (scratch_tree.size() < tree.size())
        {
            cout << "scratch_tree.size()=" << scratch_tree.size() << ". tree.size()=" << tree.size() << endl;
            throw std::runtime_error("force_tree is too small to contain scratch_tree");
        }

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

        futures.push(
            treecount1.assign(policy, device, [this](pair<gpu_uint, gpu_uint> treerange, gpu_uint parent_begin) {
                gpu_for_global(treerange.first, treerange.second, [&](gpu_uint self) {
                    gpu_uint parent = parent_begin + (self - treerange.first) / 2;

                    auto show = scratch_tree[self];
                    show.need_split = {};
                    show.is_partnode = {};

#if SURROUNDING1
                    gpu_uint totnum_old = scratch_tree[self].pend - scratch_tree[self].pbegin;
                    for (uint si = 0; si < num_surrounding(); ++si)
                    {
                        gpu_uint cousin = surrounding_link(self, si);
                        totnum_old += scratch_tree[cousin].pend - scratch_tree[cousin].pbegin;
                    }
#endif

#if SURROUNDING2
                    gpu_uint totnum0 = 0;
                    for (auto cousin : get_surrounding(self, parent, 0, true, true))
                    {
                        if (get<1>(cousin) == 0)
                        {
                            totnum0 += scratch_tree[get<0>(cousin)].pend - scratch_tree[get<0>(cousin)].pbegin;
                        }
                        else
                        {
                            totnum0 += tree[get<0>(cousin)].pend - tree[get<0>(cousin)].pbegin;
                        }
                    }

                    gpu_uint totnum1 = 0;
                    for (auto cousin : get_surrounding(self, parent, 1, true, true))
                    {
                        if (get<1>(cousin) == 0)
                        {
                            totnum1 += scratch_tree[get<0>(cousin)].pend - scratch_tree[get<0>(cousin)].pbegin;
                        }
                        else
                        {
                            totnum1 += tree[get<0>(cousin)].pend - tree[get<0>(cousin)].pbegin;
                        }
                    }

                    gpu_uint totnum2 = 0;
                    for (auto cousin : get_surrounding(self, parent, 2, true, true))
                    {
                        if (get<1>(cousin) == 0)
                        {
                            totnum2 += scratch_tree[get<0>(cousin)].pend - scratch_tree[get<0>(cousin)].pbegin;
                        }
                        else
                        {
                            totnum2 += tree[get<0>(cousin)].pend - tree[get<0>(cousin)].pbegin;
                        }
                    }

                    gpu_uint totnum3 = 0;
                    for (auto cousin : get_surrounding(self, parent, 3, true, true))
                    {
                        if (get<1>(cousin) == 0)
                        {
                            totnum3 += scratch_tree[get<0>(cousin)].pend - scratch_tree[get<0>(cousin)].pbegin;
                        }
                        else
                        {
                            totnum3 += tree[get<0>(cousin)].pend - tree[get<0>(cousin)].pbegin;
                        }
                    }
#endif

#if SURROUNDING1 && SURROUNDING2
                    gpu_if(max({ totnum_old, totnum0, totnum1, totnum2, totnum3 }) > max_nodesize)
                    {
                        gpu_assert(totnum_old == totnum0);
                        gpu_assert(totnum_old == totnum1);
                        gpu_assert(totnum_old == totnum2);
                        gpu_assert(totnum_old == totnum3);
                    }
#endif
#if SURROUNDING1
                    gpu_uint totnum = totnum_old;
#else
		gpu_uint totnum = totnum3;
#endif

                    // If true, force calculation wants to go further down the tree.
                    scratch_tree[self].need_split = (totnum > max_nodesize);

                    scratch_tree[self].is_partnode = (!scratch_tree[self].need_split && tree[parent].need_split);
                });
            }));

        futures.push(std::async(policy, [&]() {
            treecount2.assign(
                device,
                [this](pair<gpu_uint, gpu_uint> treerange,
                       gpu_uint parent_begin,
                       resource<Node_memory_request<>>& node_memory_request) {
                    Node_memory_request<gpu_uint> nmr = zero;

                    // Choosing loop order here in such a way that neighboring nodes in space will also be
                    // neighboring nodes in memory. This loop must have the same ordering in treecount2 and
                    // treecount3.
                    gpu_uint group_begin =
                        treerange.first
                        + intceil(static_cast<gpu_uint>(gpu_uint64(treerange.second - treerange.first) * group_id()
                                                        / num_groups()),
                                  gpu_uint(2));
                    gpu_uint group_end = treerange.first
                                         + intceil(static_cast<gpu_uint>(gpu_uint64(treerange.second - treerange.first)
                                                                         * (group_id() + 1) / num_groups()),
                                                   gpu_uint(2));

                    group_end = min(group_end, treerange.second);

                    gpu_for_local(group_begin, group_end, [&](gpu_uint self) {
                        gpu_uint parent = parent_begin + (self - treerange.first) / 2;

                        gpu_bool has_children;
#if SURROUNDING1
                        gpu_bool has_children1 = scratch_tree[self].need_split;
                        {
                            for (uint si = 0; si < num_surrounding(); ++si)
                            {
                                gpu_uint cousin = surrounding_link(self, si);
                                has_children1 = has_children1 || (scratch_tree[cousin].need_split);
                            }
                        }
                        has_children = has_children1;
#endif

#if SURROUNDING2
                        gpu_bool has_children2 = false;
                        for (auto cousin : get_surrounding(self, parent, 0, true, true))
                        {
                            has_children2 = has_children2 || (scratch_tree[get<0>(cousin)].need_split);
                        }
                        // If has_children is true, this node must further be split. Either because force
                        // calculation wants to continue in this node, or because it is used by another node
                        // in the vicinity.
                        has_children = has_children2;
#endif

#if SURROUNDING1 && SURROUNDING2
                        gpu_assert(has_children1 == has_children2);
#endif

                        scratch_tree[self].has_children = has_children;

                        Node_memory_request<gpu_bool> my = {
                            .split = scratch_tree[self].need_split,
                            .withchild =
                                !scratch_tree[self].need_split && has_children && !scratch_tree[self].is_partnode,
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
                       Node_memory_request<gpu_uint> memory_offset) {
                    auto old_children =
                        reinterpret<gpu_type<array<Tuint, 2>*>>(scratch.begin() + scratch_offset_old_children);
                    auto old_node_p = scratch.begin() + scratch_offset_old_node;

                    Node_memory_request<gpu_uint> next = memory_offset + this->node_memory_request[group_id()];
                    auto oldnext = next;

                    // Choosing loop order here in such a way that neighboring nodes in space will also be neighboring
                    // nodes in memory. This loop must have the same ordering in treecount2 and treecount3.
                    gpu_uint group_begin =
                        treerange.first
                        + intceil(static_cast<gpu_uint>(gpu_uint64(treerange.second - treerange.first) * group_id()
                                                        / num_groups()),
                                  gpu_uint(2));
                    gpu_uint group_end = treerange.first
                                         + intceil(static_cast<gpu_uint>(gpu_uint64(treerange.second - treerange.first)
                                                                         * (group_id() + 1) / num_groups()),
                                                   gpu_uint(2));

                    group_end = min(group_end, treerange.second);

                    const bool need_sync = (device.get_envmode() == env_CPU);

                    gpu_for_local(
                        group_begin,
                        need_sync ? (group_begin + intceil(group_end - group_begin, (gpu_uint)local_size()))
                                  : group_end,
                        [&](gpu_uint old_self) {
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
                            new_self = cond(
                                my.partnode_leaf, get_index(my.partnode_leaf, next.partnode_leaf, new_self), new_self);
                            new_self = cond(my.leaf, get_index(my.leaf, next.leaf, new_self), new_self);

                            gpu_if(old_self < group_end || !need_sync)
                            {
                                gpu_uint parent =
                                    treeranges_parent_withchild_nochild[0] + (old_self - treerange.first) / 2;
                                gpu_uint childnum = old_self % 2;
                                static_cast<scratch_treenode<gpu_T>&>(tree[new_self]) = scratch_tree[old_self];
#if GOOPAX_DEBUG
                                tree[new_self].children = {};
#endif
                                tree[new_self].parent = parent;

                                Vector<gpu_T, 3> rcenter = tree[parent].rcenter;
                                rcenter[0] += cond(childnum == 0, -halflen, halflen);
                                tree[new_self].rcenter = rot(rcenter);
                                tree[new_self].signature = tree[parent].signature * 2 + childnum;

                                gpu_assert(parent >= 2u);
                                gpu_if(parent >= 2u)
                                {
                                    gpu_array_access(tree[parent].children, childnum) = new_self;
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
        }));

        futures.push(treecount4_first_depth.assign(
            policy, device, [this](pair<gpu_uint, gpu_uint> treerange, gpu_uint treebegin_next) {
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
            }));

        for (Tuint mod3 = 0; mod3 < 3; ++mod3)
        {
            futures.push(treecount4[mod3].assign(
                policy,
                device,
                [this, mod3](pair<gpu_uint, gpu_uint> treerange,
                             gpu_uint treebegin_next,
                             gpu_uint8 depth_sublevel,
                             resource<Vector<T, 3>>& x,
                             resource<Vector<T, 3>>& v,
                             resource<T>& mass) {
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
                                        allow_preload();

                                        gpu_while(a != b)
                                        {
                                            allow_preload();
                                            gpu_if(rot(x[a], mod3)[0] >= tree[parent].rcenter[0])
                                            {
                                                gpu_break();
                                            }
                                            ++a;
                                        }
                                        gpu_while(a != b)
                                        {
                                            allow_preload();
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
                }));
        }

        futures.push(
            make_tree_clear_leaf_childs.assign(policy, device, [this](pair<gpu_uint, gpu_uint> treerange_leaf) {
                gpu_for_global(treerange_leaf.first, treerange_leaf.second, [&](gpu_uint k) {
                    tree[k].children = { 0, 0 };
                    static_cast<scratch_treenode<gpu_T>&>(tree[k]) = scratch_tree[k];
                });
            }));

        {
            auto make_surrounding_func = [this](pair<gpu_uint, gpu_uint> treerange_parent,
                                                pair<gpu_uint, gpu_uint> treerange,
                                                bool use_scratch) {
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

#if SURROUNDING2
                    array<array<gpu_uint, 6>, 2> surr_tmp;
                    for (Tuint c = 0; c < 2; ++c)
                    {
                        surr_tmp[c] = {
                            get_child(surrounding2_buf[parent][2], c),
                            get_child(surrounding2_buf[parent][3], c),
                            get_child(surrounding2_buf[parent][4], c),
                            get_child(surrounding2_buf[parent][5], c),
                            get_child(parent, 1 - c),
                            get_child(surrounding2_buf[parent][c], 1 - c),
                        };
                    }

#endif

                    for (Tuint c = 0; c < 2; ++c)
                    {
                        gpu_uint self = get_child(parent, c);
                        gpu_assert(self >= treerange.first);
                        gpu_assert(self < treerange.second);

#if SURROUNDING1
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
                        }
#endif
#if SURROUNDING2
                        surrounding2_buf[self][0] = surr_tmp[c][0];
                        surrounding2_buf[self][1] = surr_tmp[c][1];
                        surrounding2_buf[self][2] = surr_tmp[c][2];
                        surrounding2_buf[self][3] = surr_tmp[c][3];
                        surrounding2_buf[self][5 - c] = surr_tmp[c][4];
                        surrounding2_buf[self][4 + c] = surr_tmp[c][5];

                        if (!use_scratch)
                        {
                            gpu_assert(tree[surrounding2_buf[self][0]].rcenter[0] < tree[self].rcenter[0]
                                       || surrounding2_buf[self][0] < 2u);
                            gpu_assert(tree[surrounding2_buf[self][1]].rcenter[0] > tree[self].rcenter[0]
                                       || surrounding2_buf[self][1] < 2u);
                            gpu_assert(tree[surrounding2_buf[self][2]].rcenter[1] < tree[self].rcenter[1]
                                       || surrounding2_buf[self][2] < 2u);
                            gpu_assert(tree[surrounding2_buf[self][3]].rcenter[1] > tree[self].rcenter[1]
                                       || surrounding2_buf[self][3] < 2u);
                            gpu_assert(tree[surrounding2_buf[self][4]].rcenter[2] < tree[self].rcenter[2]
                                       || surrounding2_buf[self][4] < 2u);
                            gpu_assert(tree[surrounding2_buf[self][5]].rcenter[2] > tree[self].rcenter[2]
                                       || surrounding2_buf[self][5] < 2u);
                        }
#endif
                    }
                });
            };

            futures.push(
                make_surrounding.assign(policy,
                                        device,
                                        [this, make_surrounding_func](pair<gpu_uint, gpu_uint> treerange_parent,
                                                                      pair<gpu_uint, gpu_uint> treerange) {
                                            make_surrounding_func(treerange_parent, treerange, false);
                                        }));
            futures.push(make_surrounding_with_scratch.assign(
                policy,
                device,
                [this, make_surrounding_func](pair<gpu_uint, gpu_uint> treerange_parent,
                                              pair<gpu_uint, gpu_uint> treerange) {
                    make_surrounding_func(treerange_parent, treerange, true);
                }));
        }

        futures.push(
            movefunc.assign(policy, device, [this](gpu_T dt, resource<Vector<T, 3>>& v, resource<Vector<T, 3>>& x) {
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
            }));

        futures.push(kick.assign(
            policy, device, [this](gpu_T dt, const resource<Vector<T, 3>>& force, resource<Vector<T, 3>>& v) {
                gpu_for_global(0, v.size(), [&](gpu_uint k) {
                    v[k] += force[k] * dt;
                    gpu_assert(isfinite(v[k].squaredNorm()));
                });
            }));

        futures.push(
            apply_vec2.assign(policy, device, [this](const resource<Vector<T, 3>>& in, resource<Vector<T, 3>>& out) {
                apply_func(in, out);
            }));
        futures.push(apply_scalar2.assign(
            policy, device, [this](const resource<T>& in, resource<T>& out) { apply_func(in, out); }));

        for (unsigned int mod3 = 0; mod3 < 3; ++mod3)
        {
            futures.push(upwards[mod3].assign(
                policy,
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
                }));
        }

        for (bool with_potential : { false, true })
        {
            for (unsigned int mod3 = 0; mod3 < 3; ++mod3)
            {
                futures.push(handle_particles_from_multipole[with_potential][mod3].assign(
                    policy,
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

#ifndef NDEBUG
                                gpu_bool ok = true;
                                for (uint k = 0; k < 3; ++k)
                                {
                                    ok = ok
                                         && (abs((rot(x[pa], mod3) - this->tree[self].rcenter)[k])
                                             < halflen3[k] * 1.001f);
                                }
                                gpu_if(!ok)
                                {
                                    gpu_ostream DUMP(cout);
                                    DUMP << "BAD: x[" << pa << "]=" << x[pa] << ", rotated=" << rot(x[pa], mod3)
                                         << "\nv=" << this->v[pa] << "\nnode[" << self << "]=" << tree[self]
                                         << "\ndiff=" << (rot(x[pa], mod3) - this->tree[self].rcenter)
                                         << "\nhalflen3=" << halflen3 << endl;
                                }
                                gpu_assert(ok);
#endif

                                Vector<gpu_T, 3> F =
                                    rot(force_tree[self % force_tree.size()].calc_force(
                                            ((rot(x[pa], mod3) - this->tree[self].rcenter) * scale).eval()),
                                        -(Tint)mod3)
                                    * pow2(scale);

#if CALC_POTENTIAL
                                gpu_T P = force_tree[self % force_tree.size()].calc_loc_potential(
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
                    }));
            }

            futures.push(handle_particles_direct2[with_potential].assign(
                policy,
                device,
                [this, with_potential](const resource<Vector<T, 3>>& x,
                                       const resource<T>& mass,
                                       resource<Vector<T, 3>>& force,
                                       resource<T>& potential) {
                    vector<Tuint> cat = { 16, 8, 4, 3, 2, 1 };

                    vector<local_mem<pair<Tuint, Tuint>>> todo;
                    for (Tuint c : cat)
                    {
                        (void)c;
                        todo.emplace_back(local_size() * 2);
                    }
                    vector<gpu_uint> num_todo(cat.size(), 0);

                    auto doit = [&](Tuint c) {
                        auto [self, pbegin] = todo[c][num_todo[c] + local_id()];
                        gpu_uint N = (tree[self].pend - pbegin);
                        if (c == 0)
                        {
                            N = min(N, cat[c]);
                        }
                        Tuint min_N = 1;
                        if (c + 1 < cat.size())
                        {
                            min_N = cat[c + 1] + 1;
                            assert(cat[c] <= 2 * cat[c + 1]);
                        }
                        if (cat[c] == min_N)
                        {
                            gpu_assert(N == cat[c]);
                            N = cat[c];
                        }

                        gpu_assert(N <= cat[c]);
                        vector<Vector<gpu_T, 3>> my_x(cat[c]);
                        for (Tuint k = 0; k < cat[c]; ++k)
                        {
                            gpu_if(k < N || k < min_N)
                            {
                                my_x[k] = x[pbegin + k];
                            }
                        }

#if SURROUNDING1
                        vector<Vector<gpu_T, 3>> F1(cat[c], { 0, 0, 0 });
                        vector<gpu_T> P1(cat[c], 0);
                        {
                            // Starting with the self node, since it is not included in the surrounding list.
                            gpu_int si = -1;
                            gpu_uint other_node = self; // surrounding_link(self, si);
                            gpu_uint other_p = tree[other_node].pbegin;
                            gpu_uint other_pend = tree[other_node].pend;

                            gpu_bool docontinue = true;
                            gpu_while(docontinue)
                            {
                                for (Tuint k = 0; k < cat[c]; ++k)
                                {
                                    const Vector<gpu_T, 3> dist = x[other_p] - my_x[k];
                                    F1[k] += dist
                                             * (mass[other_p] * pow<-1, 2>(dist.squaredNorm() + 1E-20f)
                                                * pow2(pow<-1, 2>(dist.squaredNorm() + 1E-20f)));

                                    P1[k] += cond(dist.squaredNorm() == 0,
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
                        }
#endif

#if SURROUNDING2
                        vector<Vector<gpu_T, 3>> F2(cat[c], { 0, 0, 0 });
                        vector<gpu_T> P2(cat[c], 0);
                        {
                            auto vicinity = get_surrounding(self, tree[self].parent, 3, true, false);
                            private_mem<pair<Tuint, Tuint>> ranges(vicinity.size());
                            for (Tuint k = 0; k < vicinity.size(); ++k)
                            {
                                ranges[k] = { tree[get<0>(vicinity[k])].pbegin, tree[get<0>(vicinity[k])].pend };
                            }

                            // Starting with the self node, since it is not included in the surrounding list.
                            // gpu_int si = -1;
                            // gpu_uint other_node = vicinity[0];
                            gpu_uint range_i = 0;
                            gpu_uint other_p = ranges[0].first;
                            gpu_uint other_pend = ranges[0].second;

                            gpu_bool docontinue = true;
                            gpu_while(docontinue)
                            {
                                for (Tuint k = 0; k < cat[c]; ++k)
                                {
                                    const Vector<gpu_T, 3> dist = x[other_p] - my_x[k];
                                    F2[k] += dist
                                             * (mass[other_p] * pow<-1, 2>(dist.squaredNorm() + 1E-20f)
                                                * pow2(pow<-1, 2>(dist.squaredNorm() + 1E-20f)));

                                    P2[k] += cond(dist.squaredNorm() == 0,
                                                  0.f,
                                                  -(mass[other_p] * pow<-1, 2>(dist.squaredNorm() + 1E-20f)));
                                }
                                ++other_p;
                                gpu_while(other_p == other_pend)
                                {
                                    ++range_i;
                                    gpu_if(range_i == vicinity.size())
                                    {
                                        docontinue = false;
                                        gpu_break();
                                    }
                                    other_p = ranges[range_i].first;
                                    other_pend = ranges[range_i].second;
                                }
                            }
                        }
#endif

                        for (Tuint k = 0; k < cat[c]; ++k)
                        {
#if SURROUNDING1 && SURROUNDING2
                            gpu_assert((F1[k] - F2[k]).squaredNorm() < 1.f);
                            gpu_assert(abs(P1[k] - P2[k]) < 1E-5f);
#endif

#if SURROUNDING1
                            Vector<gpu_T, 3> F = F1[k];
                            gpu_T P = P1[k];
#else
                            Vector<gpu_T, 3> F = F2[k];
                            gpu_T P = P2[k];
#endif

                            gpu_if(k < N || k < min_N)
                            {
                                force[pbegin + k] = F;
                                if (with_potential)
                                {
                                    potential[pbegin + k] = P;
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

                            for (Tuint c = 0; c < cat.size(); ++c)
                            {
                                auto run_c = [&]() {
                                    gpu_bool yes;
                                    if (c + 1 < cat.size())
                                    {
                                        yes = gpu_int(pend - pbegin) > (int)cat[c + 1];
                                    }
                                    else
                                    {
                                        yes = gpu_int(pend - pbegin) > 0;
                                    }
                                    gpu_uint i = get_index(yes, num_todo[c]);

                                    gpu_if(yes)
                                    {
                                        todo[c][i] = { self, pbegin };
                                        pbegin += cat[c];
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
                                    gpu_while(work_group_any(gpu_int(pend - pbegin) > (int)cat[0] + (int)cat[1]))
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
                    for (Tuint c = 0; c < cat.size(); ++c)
                    {
                        gpu_if(local_id() < num_todo[c])
                        {
                            num_todo[c] = 0;
                            doit(c);
                        }
                    }
                }));

            futures.push(downwards2[with_potential].assign(
                policy, device, [this, with_potential](pair<gpu_uint, gpu_uint> split_range) {
                    gpu_for_global(split_range.first, split_range.second, [&](gpu_uint parent) {
                        gpu_assert(this->tree[parent].need_split);

                        const auto parent_treenode = force_tree[parent % force_tree.size()];

#if SURROUNDING1
                        array<gpu_force_multipole_f32, 2> new_multipole_f32_1 = zero;
                        array<gpu_force_multipole_bf16, 2> new_multipole_bf16_1 = zero;
                        Tuint count1 = 0;
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

                                        matter_tree[cousin].makelocal(vdata.make_real(-cousinvec, 2 * pow<1, 3>(2.0)),
                                                                      &new_multipole_f32_1[c],
                                                                      &new_multipole_bf16_1[c]);

                                        ++count1;
                                    }
                                }
                            }
                        }

                        assert(count1 == (this->num_surrounding() + 1) * 2);
#endif
#if SURROUNDING2
                        array<gpu_force_multipole_f32, 2> new_multipole_f32_2 = zero;
                        array<gpu_force_multipole_bf16, 2> new_multipole_bf16_2 = zero;

                        Tuint count2 = 0;
                        for (auto uncle : get_surrounding(parent, tree[parent].parent, 0, true, false))
                        {
                            for (int c = 0; c < 2; ++c)
                            {
                                for (int c2 = 0; c2 < 2; ++c2)
                                {
                                    Vector<Tint, 3> cousinvec = vdata.parent2child(get<2>(uncle), c2 - c);
                                    if (!vdata.local_indices.contains(vdata.make_index(cousinvec))
                                        && cousinvec.squaredNorm() != 0)
                                    {
                                        gpu_uint cousin = this->tree[get<0>(uncle)].children[c2];

                                        matter_tree[cousin].makelocal(vdata.make_real(-cousinvec, 2 * pow<1, 3>(2.0)),
                                                                      &new_multipole_f32_2[c],
                                                                      &new_multipole_bf16_2[c]);

                                        ++count2;
                                    }
                                }
                            }
                        }
                        assert(count2 == (this->num_surrounding() + 1) * 2);
#endif

#if SURROUNDING1 && SURROUNDING2
                        for (int c = 0; c < 2; ++c)
                        {
                            gpu_assert((new_multipole_f32_1[c].calc_force(Vector<double, 3>{ 0.7, 0.5, -0.8 })
                                        - new_multipole_f32_2[c].calc_force(Vector<double, 3>{ 0.7, 0.5, -0.8 }))
                                           .squaredNorm()
                                       < 1E-5f);
                        }
#endif

                        for (int c = 0; c < 2; ++c)
                        {
#if SURROUNDING1
                            gpu_force_multipole new_multipole =
                                static_cast<gpu_force_multipole>(new_multipole_bf16_1[c]);
                            new_multipole += static_cast<gpu_force_multipole>(new_multipole_f32_1[c]);
#else
                        gpu_force_multipole new_multipole = static_cast<gpu_force_multipole>(new_multipole_bf16_2[c]);
                        new_multipole += static_cast<gpu_force_multipole>(new_multipole_f32_2[c]);
#endif

                            Vector<Tdouble, 3> shiftvec = { 0, 0, pow<1, 3>(2.0) * (2 * c - 1) };
                            new_multipole += parent_treenode.rot().scale_loc(pow<1, 3>(2.0)).shift_loc(shiftvec);

                            if (!with_potential)
                            {
                                // Marking the first order multipole as unused. This should speed up the
                                // calculations if the potential is not required.
                                new_multipole.template get<0>().A[0] = {};
                            }

                            gpu_uint self = this->tree[parent].children[c];
                            force_tree[self % force_tree.size()] = new_multipole;
                        }
                    });
                }));
            cout << "created downwards [with_potential=" << with_potential << "]" << endl;
        }

        futures.push(update_tree_set_nodelink.assign(policy, device, [this]() {
            auto nodelink = scratch.begin() + scratch_offset_nodelink;

            multi_level_loop(all_leaf_ranges_packed_buf, global_id(), global_size(), [&](gpu_uint self) {
                gpu_for(this->tree[self].pbegin, this->tree[self].pend, [&](gpu_uint p) { nodelink[p] = self; });
            });
        }));

        futures.push(update_tree_1.assign(policy, device, [this](const resource<Vector<T, 3>>& x) {
            auto next_p = scratch.begin() + scratch_offset_next_p;
            auto particle_list = scratch.begin() + scratch_offset_particle_list;
            auto nodelink = scratch.begin() + scratch_offset_nodelink;

            gpu_for_global(0, x.size(), [&](gpu_uint p) {
                gpu_uint node = nodelink[p];
                gpu_uint depth = this->tree[node].depth;
                gpu_signature_t new_sig = calc_sig<signature_t>(x[p], MAX_DEPTH());

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

                    gpu_uint child = gpu_array_access(this->tree[i].children, (new_sig & bit) != 0);
                    // Going down to the new leaf.
                    gpu_while(child != 0)
                    {
                        i = child;
                        bit >>= 1;
                        child = gpu_array_access(this->tree[i].children, (new_sig & bit) != 0);
                    }

                    // Add to linked list of particles that are to be added to node i.
                    particle_list[p] = atomic_xchg(next_p[i], p, std::memory_order_relaxed);
                }
            });
        }));

        futures.push(update_tree_upwards.assign(
            policy, device, [this](pair<gpu_uint, gpu_uint> leaf_range, pair<gpu_uint, gpu_uint> withchild_range) {
                auto next_p = scratch.begin() + scratch_offset_next_p;
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
                });
            }));

        futures.push(update_tree_downwards.assign(policy, device, [this](pair<gpu_uint, gpu_uint> treerange_parent) {
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
                }
            });
        }));

        const Tuint scratch_end = scratch_offset_old_node + tree.size();

        cout << "scratch usage: " << scratch_end << " / " << scratch.size() << endl;
        if (scratch_end > scratch.size())
        {
            cout << "scratch size to small." << endl;
            exit(1);
        }

        futures.push(update_tree_reorder_particles.assign(policy, device, [this]() {
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
        }));

        cout << "Waiting for kernel creations" << endl;

        while (!futures.empty())
        {
            futures.top().get();
            futures.pop();
        }
    }
};
