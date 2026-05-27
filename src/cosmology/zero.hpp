#pragma once
#include <goopax>

#if __cpp_concepts >= 201907

struct zero_initializer
{
    operator int() const
    {
        return 0;
    }
    template<typename T>
    operator goopax::gpu_type<T>() const
    {
        return 0;
    }

    template<typename T>
        requires(!std::is_convertible_v<int, T>)
    operator T() const;
};

namespace zero_checks
{
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
struct has_Constant : false_type
{
};

template<typename T>
    requires(std::is_convertible_v<decltype(T::Constant(zero_initializer())), T>)
struct has_Constant<T> : true_type
{
};

template<typename T>
struct has_zero : false_type
{
};

template<typename T>
    requires(std::is_convertible_v<decltype(T::zero()), T>)
struct has_zero<T> : true_type
{
};
}

template<typename T>
    requires(!std::is_convertible_v<int, T>)
zero_initializer::operator T() const
{
    if constexpr (std::is_convertible<int, T>::value)
    {
        return 0;
    }
    else if constexpr (zero_checks::has_Constant<T>::value)
    {
        return T::Constant(*this);
    }
    else if constexpr (zero_checks::has_zero<T>::value)
    {
        return T::zero();
    }
    else if constexpr (std::ranges::input_range<T>)
    {
        T ret;
        std::fill(ret.begin(), ret.end(), *this);
        return ret;
    }
    else
    {
        static_assert(false, "Cannot find appropriate method to set to zero.");
    }
}

static constexpr zero_initializer zero;

#endif
