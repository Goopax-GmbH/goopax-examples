#include <goopax>

template<typename T, size_t N>
struct gpu_array : public std::array<T, N>
{
    using std::array<T, N>::operator[];

    template<typename uint_type>
        requires(!std::is_convertible_v<uint_type, size_t>)
    T& operator[](uint_type k)
    {
        return gpu_array_access(*this, k);
    }
    template<typename uint_type>
        requires(!std::is_convertible_v<uint_type, size_t>)
    const T& operator[](uint_type k) const
    {
        return gpu_array_access(*this, k);
    }

    gpu_array(const std::array<T, N>& a)
        : std::array<T, N>(a)
    {
    }
    gpu_array() = default;

    using goopax_struct_type = T;
    template<typename X>
    using goopax_struct_changetype = gpu_array<typename goopax::goopax_struct_changetype<T, X>::type, N>;
};
