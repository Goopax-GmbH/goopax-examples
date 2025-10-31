#include <goopax_extra/output.hpp>

template<typename S>
concept ostream_type = std::same_as<S, std::ostream> || std::same_as<S, goopax::gpu_ostream>;

template<ostream_type S, typename A, typename B>
S& operator<<(S& s, const std::pair<A, B>& p);

template<ostream_type S, typename... A>
S& operator<<(S& s, const std::tuple<A...>& p);

template<ostream_type S, typename V>
#if __cpp_lib_ranges >= 201911
    requires std::ranges::input_range<V>
#else
    requires std::is_convertible<typename std::iterator_traits<typename V::iterator>::iterator_category,
                                 std::input_iterator_tag>::value
#endif
             && std::is_class<V>::value && (!std::is_same<V, std::string>::value)
             && (std::is_same<S, std::ostream>::value || std::is_same<S, goopax::gpu_ostream>::value)
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

template<size_t pos, ostream_type S, typename T>
void print_tuple_entries(S& s, const T& t)
{
    if (pos != 0)
    {
        s << ", ";
    }
    s << get<pos>(t);
    if constexpr (pos + 1 < std::tuple_size<T>::value)
    {
        print_tuple_entries<pos + 1>(s, t);
    }
}

template<ostream_type S, typename... A>
S& operator<<(S& s, const std::tuple<A...>& t)
{
    s << "[";
    print_tuple_entries<0>(s, t);
    s << "]";
    return s;
}

template<ostream_type S, typename A, typename B>
S& operator<<(S& s, const std::pair<A, B>& p)
{
    return s << std::make_tuple(p.first, p.second);
}
