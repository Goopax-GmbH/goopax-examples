#pragma once

template<typename T>
struct has_zero : false_type
{
};

template<typename T>
    requires(std::is_convertible_v<decltype(T::Zero()), T>)
struct has_zero<T> : true_type
{
};

template<typename T>
    requires(std::is_convertible_v<decltype(T::zero()), T>)
struct has_zero<T> : true_type
{
};

struct zero_initializer;

struct zero_initializer
{
    template<typename T>
        requires(std::is_convertible_v<int, T>
                 && (is_same_v<typename goopax::get_gpu_mode<T>::type, int>
                     || is_same_v<typename goopax::get_gpu_mode<T>::type, goopax::debugtype<int>>
                     || is_same_v<typename goopax::get_gpu_mode<T>::type, goopax::gpu_int>))
    operator T() const
    {
        return static_cast<T>(0);
    }

    template<typename T>
#if defined(__clang__)
        requires(!std::is_convertible_v<int, T> && std::is_convertible_v<decltype(T::Constant({})), T>)
#else
        requires(!std::is_convertible_v<int, T> && std::is_convertible_v<decltype(T::Constant(zero_initializer())), T>)
#endif
    operator T() const
    {
        return T::Constant(zero_initializer());
    }

    template<typename T>
        requires(!std::is_convertible_v<int, T> && std::is_convertible_v<decltype(T::zero()), T>)
    operator T() const
    {
        return T::zero();
    }

    template<typename T>
        requires(!has_zero<T>::value) && (!std::is_convertible_v<int, T>)
    operator T() const;
};

template<typename T>
struct has_fill : false_type
{
};

template<typename T>
    requires(is_same<decltype(std::declval<T>().fill(zero_initializer())), void>::value)
struct has_fill<T> : true_type
{
};

template<typename T>
    requires(!has_zero<T>::value) && (!std::is_convertible_v<int, T>)
zero_initializer::operator T() const
{
    T ret;
    if constexpr (has_fill<T>::value)
    {
        ret.fill(zero_initializer());
    }
    else
    {
        fill(ret, zero_initializer());
    }
    return ret;
}

static constexpr zero_initializer zero;
