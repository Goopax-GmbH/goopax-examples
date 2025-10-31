#include <goopax_extra/struct_types.hpp>

template<class A, class B, class S>
S& operator<<(S& s, const pair<A, B>& p)
{
    return s << "<" << p.first << "|" << p.second << ">;";
}

#define GOOPAX_PREPARE_STRUCT2(NAME, X) \
    using goopax_struct_type = X;       \
    template<typename XX>               \
    using goopax_struct_changetype = NAME<typename goopax_struct_changetype<X, XX>::type>;

namespace heapsort
{
template<typename RES, typename CMP>
void siftdown(RES& a, const gpu_uint start, const gpu_uint end, CMP cmp)
{
    gpu_uint root = start;
    gpu_while(root * 2 + 1 <= end)
    {
        gpu_uint child = root * 2 + 1;
        gpu_uint swap = root;
        swap = cond(cmp(a[swap], a[child]), child, swap);
        swap = cond(cmp(a[swap], a[cond(child + 1 <= end, child + 1, swap)]), child + 1, swap);

        gpu_if(swap == root)
        {
            gpu_break();
        }
        std::swap(a[root], a[swap]);
        root = swap;
    }
}

template<typename FUNC, typename CMP>
void heapify(FUNC& a, const gpu_uint count, CMP cmp)
{
    gpu_for<std::greater_equal<>>((count - 2) / 2, 0, -1, [&](gpu_int start) { siftdown(a, start, count - 1, cmp); });
}

template<typename RES, typename CMP = std::less<>>
void heapsort(RES& a, const gpu_uint count, CMP cmp = std::less<>())
{
    heapify(a, count, cmp);
    gpu_for<std::greater<>>(count - 1, 0, -1, [&](gpu_int end) {
        std::swap(a[end], a[0]);
        siftdown(a, 0, end - 1, cmp);
    });
}

template<typename T, typename CMP = std::less<>>
void sort_tiny(gpu_type<T*> data, gpu_uint size, unsigned int max_size, CMP cmp = std::less<>())
{
    private_mem<T> tmp(max_size);

    gpu_for(0, size, [&](gpu_uint k) { tmp[k] = data[k]; });
    heapsort(tmp, size, cmp);
    gpu_for(0, size, [&](gpu_uint k) { data[k] = tmp[k]; });
}

}

template<typename T, typename key_t>
struct radix_sort
{
    using gpu_T = typename make_gpu<T>::type;
    using gpu_key_t = typename make_gpu<key_t>::type;

    std::function<gpu_bool(gpu_T, gpu_T)> cmp;
    std::function<gpu_key_t(T)> get_key;
    const Tuint ls_use;
    const Tuint gs_use;
    const Tuint ng_use = gs_use / ls_use;
    buffer<pair<Tuint, Tuint>> ranges;
    buffer<Tuint> local_offsets;
    buffer<Tuint> group_offsets;
    buffer<Tuint> key_offsets;

    Tuint bigrange_bits;
    Tuint smallrange_bits;

    template<class X = Tuint>
    struct smallrange_info
    {
        GOOPAX_PREPARE_STRUCT2(smallrange_info, X)

        using uint_type = typename change_gpu_mode<unsigned int, X>::type;

        uint_type begin;
        uint_type end;
        uint_type bits;

        smallrange_info()
        {
        }

        smallrange_info(uint_type begin0, uint_type end0, uint_type bits0)
            : begin(begin0)
            , end(end0)
            , bits(bits0)
        {
        }

        template<class OSTREAM>
        friend OSTREAM& operator<<(OSTREAM& s, const smallrange_info& p)
        {
            return s << "[range=" << p.begin << " ... " << p.end << ", bits=" << p.bits << "]";
        }
    };
    buffer<smallrange_info<>> smallrange;

    kernel<void(const buffer<T>& src,
                const buffer<pair<Tuint, Tuint>>& ranges,
                const Tuint num_ranges,
                const Tuint shift,
                buffer<Tuint>& local_offset,
                buffer<Tuint>& group_count)>
        radix_sort_func1;

    kernel<void(buffer<Tuint>& group_count, buffer<Tuint>& key_count, Tuint num_ranges)> radix_addfunc1;

    kernel<void(buffer<Tuint>& key_offsets, const buffer<pair<Tuint, Tuint>>& ranges, Tuint num_ranges)> radix_addfunc2;

    kernel<void(const buffer<T>& src,
                const buffer<pair<Tuint, Tuint>>& ranges,
                const Tuint num_ranges,
                const buffer<Tuint>& local_offsets,
                const buffer<Tuint>& group_offsets,
                const buffer<Tuint>& key_offsets,
                const Tuint shift,
                buffer<T>& dest)>
        radix_writefunc;

    kernel<void(buffer<T>& src, buffer<T>& tmp, buffer<smallrange_info<>>& smallrange, const Tuint smallrange_size)>
        smallsortfunc;

#ifndef NDEBUG
    kernel<void(const buffer<T>& p, Tuint size)> testsortfunc;
#endif

    void operator()(buffer<T>& plist1, buffer<T>& plist2, const Tuint max_depthbits)
    {
        goopax_device device = plist1.get_device();
        const unsigned int bits = bigrange_bits;

        vector<pair<Tuint, Tuint>> bigrangevec;
        bigrangevec.reserve(this->ranges.size());
        bigrangevec.assign({ { 0, plist1.size() } });

        vector<smallrange_info<>> smallrangevec;
        smallrangevec.reserve(this->smallrange.size());

#if WITH_TIMINGS
        device.wait_all();
        auto t0 = steady_clock::now();
#endif

        for (Tint shift_i = Tint(max_depthbits) - bits; shift_i >= -Tint(bits) + 1; shift_i -= bits)
        {
            Tuint shift = max(shift_i, 0);
            // cout << "\nshift=" << shift << endl;

            if (bigrangevec.empty())
                break;

            if (ranges.size() < bigrangevec.size())
            {
                Tsize_t newsize = bigrangevec.size() * 1.1;
                // cout1 << "Increasing bigrange size to " << newsize << endl;
                ranges.assign(device, newsize);
                local_offsets.assign(device, newsize * (1 << bits) * gs_use);
                group_offsets.assign(device, newsize * (1 << bits) * ng_use);
                key_offsets.assign(device, newsize * (1 << bits));
            }
            {
                buffer_map<pair<Tuint, Tuint>> ranges(this->ranges);
                for (Tsize_t k = 0; k < bigrangevec.size(); ++k)
                {
                    ranges[k] = bigrangevec[k];
                }
            }

            radix_sort_func1(plist1, ranges, bigrangevec.size(), shift, local_offsets, group_offsets);
            radix_addfunc1(group_offsets, key_offsets, bigrangevec.size());

            const Tsize_t old_bigrangevecsize = bigrangevec.size();
            {
                const_buffer_map<Tuint> key_offsets(this->key_offsets);

                vector<pair<Tuint, Tuint>> newbigrangevec;
                const Tsize_t max_size = max(plist1.size() / (2 * ng_use), (Tuint)256);
                for (Tuint r = 0; r < bigrangevec.size(); ++r)
                {
                    Tuint begin = bigrangevec[r].first;
                    for (Tuint key = 0; key < (1u << bits); ++key)
                    {
                        auto size = key_offsets[r * (1 << bits) + key];
                        if (size > max_size)
                        {
                            newbigrangevec.push_back({ begin, begin + size });
                        }
                        else if (size >= 1)
                        {
                            smallrangevec.push_back({ begin, begin + size, shift });
                        }
                        begin += size;
                    }
                }
                bigrangevec = std::move(newbigrangevec);
            }

            // cout << "bigrange=" << bigrangevec << endl;
            // cout << "smallrange=" << smallrangevec << endl;

            radix_addfunc2(key_offsets, ranges, old_bigrangevecsize);

            if (!smallrangevec.empty())
            {
                plist2.copy(plist1);
            }

            radix_writefunc(
                plist1, ranges, old_bigrangevecsize, local_offsets, group_offsets, key_offsets, shift, plist2);

            // cout1 << "Swapping" << endl;
            swap(plist1, plist2);
        }

#if WITH_TIMINGS
        device.wait_all();
        auto t1 = steady_clock::now();
#endif

        enum
        {
            max_bits_hardlimit = 8
        };
        Tsize_t maxsize = smallrangevec.size()
                          + ng_use * ((1 << max_bits_hardlimit) - 1)
                                * ((max_depthbits + max_bits_hardlimit - 1) / max_bits_hardlimit);
        // maxsize += ((1<<smallsort::bits)-1) * (max_depthbits / radix_sort::smallsort::bits) * num_groups();
        if (smallrange.size() < maxsize)
        {
            smallrange = buffer<smallrange_info<>>(device, maxsize * 1.1);
            // cout1 << "Increasing smallrange size to " << smallrange.size() << endl;
        }
        {
            buffer_map smallrange(this->smallrange);
            std::copy(smallrangevec.begin(), smallrangevec.end(), smallrange.begin());
        }
        smallsortfunc(plist1, plist2, smallrange, smallrangevec.size());

        //    cout1 << "plist1=" << plist1 << endl;
#if WITH_TIMINGS
        device.wait_all();
        auto t2 = steady_clock::now();
#endif

#if WITH_TIMINGS
        cout << "bigrange: " << duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << endl;
        cout << "smallrange: " << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << endl;
#endif

#ifndef NDEBUG
        testsortfunc(plist1, plist1.size());
#endif
    }

    radix_sort(goopax_device device, std::function<gpu_bool(gpu_T, gpu_T)> cmp0)
        : cmp(cmp0)
        , ls_use(device.default_local_size())
        , gs_use(device.default_global_size_min())
        , ranges(device, 0)
        , local_offsets(device, 0)
        , group_offsets(device, 0)
        , key_offsets(device, 0)
        , smallrange(device, 0)
    {
        unsigned int num_registers = device.max_registers();
        if (num_registers == 0)
            num_registers = 128;
        // cout << "num_registers=" << num_registers << endl;
        // cout << "sizeof(T)=" << sizeof(T) << endl;

        {
            Tuint max_bits = 2;
            while ((1 << (max_bits + 1)) * sizeof(T) / sizeof(float) < num_registers * 0.7)
            {
                ++max_bits;
            }

            ++max_bits;
            // cout << "Using max_bits=" << max_bits << endl;
            smallrange_bits = max_bits;
            bigrange_bits = max_bits;
        }

        radix_sort_func1.assign(
            device,
            [this](const resource<T>& src,
                   const resource<pair<Tuint, Tuint>>& ranges,
                   const gpu_uint num_ranges,
                   const gpu_uint shift,
                   resource<Tuint>& local_offset,
                   resource<Tuint>& group_count) {
                gpu_for(0, num_ranges, [&](gpu_uint r) {
                    private_mem<Tuint> localcount(1 << bigrange_bits);
                    for (Tuint k = 0; k < (1u << bigrange_bits); ++k)
                    {
                        localcount[k] = 0;
                    }

                    const gpu_uint begin = ranges[r].first;
                    const gpu_uint end = ranges[r].second;

                    gpu_for_global(begin, end, [&](gpu_uint k) {
                        gpu_uint key = gpu_uint(src[k].first >> shift) & ((1 << bigrange_bits) - 1);
                        ++localcount[key];
                    });
                    gpu_for(0, (1 << bigrange_bits), [&](gpu_uint key) {
                        gpu_uint my_offset = work_group_scan_exclusive_add(localcount[key]);
                        local_offset[r * (1 << bigrange_bits) * global_size() + key * global_size() + global_id()] =
                            my_offset;

                        gpu_if(local_id() == local_size() - 1)
                        {
                            group_count[r * (1 << bigrange_bits) * num_groups() + key * num_groups() + group_id()] =
                                my_offset + localcount[key];
                        }
                    });
                });
            },
            ls_use,
            gs_use);

        radix_addfunc1.assign(
            device,
            [this](resource<Tuint>& group_count, resource<Tuint>& key_count, gpu_uint num_ranges) {
                gpu_for_group(0, (1 << bigrange_bits) * num_ranges, [&](gpu_uint keyrange) {
                    gpu_uint key = keyrange % (1 << bigrange_bits);
                    gpu_uint r = keyrange / (1 << bigrange_bits);

                    gpu_uint sum = 0;
                    gpu_for_local(0, intceil(num_groups(), local_size()), [&](gpu_uint t) {
                        gpu_uint pos = r * (1 << bigrange_bits) * num_groups() + key * num_groups() + t;
                        gpu_uint val = 0;
                        gpu_if(t < num_groups())
                        {
                            val = group_count[pos];
                        }
                        gpu_uint val_offset = work_group_scan_exclusive_add(val, local_size());
                        gpu_if(t < num_groups())
                        {
                            group_count[pos] = sum + val_offset;
                        }
                        sum += shuffle(val_offset + val, local_size() - 1, local_size());
                    });
                    gpu_if(local_id() == 0)
                    {
                        key_count[(1 << bigrange_bits) * r + key] = sum;
                    }
                });
            },
            ls_use,
            gs_use);

        radix_addfunc2.assign(
            device,
            [this](resource<Tuint>& key_offsets, const resource<pair<Tuint, Tuint>>& ranges, gpu_uint num_ranges) {
                gpu_for_global(0, num_ranges, [&](gpu_uint r) {
                    gpu_uint sum = ranges[r].first;
                    gpu_for(0, (1 << bigrange_bits), [&](gpu_uint key) {
                        gpu_uint pos = r * (1 << bigrange_bits) + key;
                        gpu_uint val = key_offsets[pos];
                        key_offsets[pos] = sum;
                        sum += val;
                    });
                });
            });

        radix_writefunc.assign(
            device,
            [this](const resource<T>& src,
                   const resource<pair<Tuint, Tuint>>& ranges,
                   const gpu_uint num_ranges,
                   const resource<Tuint>& local_offsets,
                   const resource<Tuint>& group_offsets,
                   const resource<Tuint>& key_offsets,
                   const gpu_uint shift,
                   resource<T>& dest) {
                private_mem<Tuint> offsets(1 << bigrange_bits);
                local_mem<Tuint> thisgroup_offsets(1 << bigrange_bits);
                gpu_for(0, num_ranges, [&](gpu_uint r) {
                    gpu_for_local(0, (1 << bigrange_bits), [&](gpu_uint key) {
                        thisgroup_offsets[key] =
                            key_offsets[r * (1 << bigrange_bits) + key]
                            + group_offsets[r * (1 << bigrange_bits) * num_groups() + key * num_groups() + group_id()];
                    });
                    thisgroup_offsets.barrier();
                    gpu_for(0, (1 << bigrange_bits), [&](gpu_uint key) {
                        offsets[key] = (local_offsets[r * (1 << bigrange_bits) * global_size() + key * global_size()
                                                      + global_id()])
                                       + thisgroup_offsets[key];
                    });
                    thisgroup_offsets.barrier();

                    const gpu_uint begin = ranges[r].first;
                    const gpu_uint end = ranges[r].second;
                    gpu_for_global(begin, end, [&](gpu_uint k) {
                        gpu_uint key = gpu_uint(src[k].first >> shift) & ((1 << bigrange_bits) - 1);
                        gpu_uint pos = offsets[key]++;
                        dest[pos] = src[k];
                    });
                });
            },
            ls_use,
            gs_use);

        smallsortfunc.assign(
            device,
            [this](resource<T>& src,
                   resource<T>& tmp,
                   resource<smallrange_info<>>& smallrange,
                   const gpu_uint smallrange_size) {
                smallrange_info<gpu_uint> myrange;

                gpu_uint num_tiny = 0;

                gpu_uint smallrange_end = (smallrange_size + (num_groups() - 1) - group_id()) / num_groups();

                gpu_while(smallrange_end != 0)
                {
                    --smallrange_end;
                    const smallrange_info<gpu_uint> range = smallrange[smallrange_end * num_groups() + group_id()];
                    gpu_if((range.end - range.begin <= (1u << smallrange_bits)))
                    {
                        gpu_if(range.end - range.begin >= 2u)
                        {
                            myrange.begin = cond(num_tiny == local_id(), range.begin, myrange.begin);
                            myrange.end = cond(num_tiny == local_id(), range.end, myrange.end);
                            myrange.bits = cond(num_tiny == local_id(), range.bits, myrange.bits);
                            ++num_tiny;
                            gpu_if(num_tiny == local_size())
                            {
                                src.barrier();
                                heapsort::sort_tiny(src.begin() + myrange.begin,
                                                    myrange.end - myrange.begin,
                                                    (1 << smallrange_bits),
                                                    cmp);
                                src.barrier();
                                num_tiny = 0;
                            }
                        }
                    }
                    gpu_else
                    {
                        local_barrier();
                        private_mem<Tuint> count(1 << smallrange_bits);
                        const gpu_uint bits = min(32 - countl_zero(range.end - range.begin) + 1, smallrange_bits);
                        gpu_for(0, (1u << bits), [&](gpu_uint key) { count[key] = 0; });

                        gpu_for_local(range.begin, range.end, [&](gpu_uint k) {
                            gpu_uint key = gpu_uint(src[k].first >> (max(range.bits, bits) - bits))
                                           & ((1u << bits) - 1); // FIXME: Make range.bits%bits==0.
                            gpu_assert(bits <= 32u);
                            ++count[key];
                        });

                        gpu_uint sum = range.begin;
                        gpu_for(0, (1u << bits), [&](gpu_uint key) {
                            gpu_uint offset = work_group_scan_exclusive_add(count[key]);
                            gpu_uint total = shuffle(offset + count[key], local_size() - 1, local_size());
                            count[key] = offset + sum;

                            gpu_if(range.bits > bits && local_id() == 0 && total >= 2u)
                            {
                                smallrange[(smallrange_end++) * num_groups() + group_id()] =
                                    smallrange_info<gpu_uint>(sum, sum + total, range.bits - bits);
                            }

                            sum += total;
                        });

                        gpu_for_local(range.begin, range.end, [&](gpu_uint k) {
                            gpu_uint key = gpu_uint(src[k].first >> (max(range.bits, bits) - bits))
                                           & ((1u << bits) - 1); // FIXME: Make range.bits%bits==0.
                            gpu_uint pos = count[key]++;
                            tmp[pos] = src[k];
                        });
                        smallrange.barrier();
                        tmp.barrier();
                        src.barrier();
                        gpu_for_local(range.begin,
                                      range.end,
                                      [&](gpu_uint k) // FIXME: swap somehow.
                                      { src[k] = tmp[k]; });

                        smallrange_end = shuffle(smallrange_end, 0, local_size());
                    }
                }
                src.barrier();
                gpu_if(local_id() < num_tiny)
                {
                    heapsort::sort_tiny(
                        src.begin() + myrange.begin, myrange.end - myrange.begin, (1 << smallrange_bits), cmp);
                }
            },
            ls_use,
            gs_use);

#ifndef NDEBUG
        testsortfunc.assign(device, [](const resource<T>& p, gpu_uint size) {
            gpu_for_global(0, size - 1, [&](gpu_uint k) { gpu_assert(p[k].first <= p[k + 1].first); });
        });
#endif
    }
};
