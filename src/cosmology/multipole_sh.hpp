template<typename TO, typename FROM>
complex<TO> cast(const complex<FROM>& x)
{
    return complex<TO>{ static_cast<TO>(x.real()), static_cast<TO>(x.imag()) };
}

template<typename T>
gpu_ostream& operator<<(gpu_ostream& s, const complex<gpu_type<T>>& c)
{
    return s << "[" << c.real() << "," << c.imag() << "]";
}

template<class T, unsigned int N>
struct multipole
{
    using cpu_T = typename make_cpu<T>::type;

    using goopax_struct_type = T;
    template<typename X>
    using goopax_struct_changetype = multipole<typename goopax_struct_changetype<T, X>::type, N>;
    using uint_type = typename change_gpu_mode<unsigned int, T>::type;

    static constexpr size_t datasize = (N + 1) * (N + 1);

    Vector<T, datasize> coeffs;

    multipole()
    {
    }

    static multipole zero()
    {
        multipole ret;
        ret.coeffs.fill(0);
        return ret;
    }

    multipole& operator+=(const multipole& b)
    {
        coeffs += b.coeffs;
        return *this;
    }

    template<class U>
    multipole(const multipole<U, N>& b)
    {
        coeffs = b.coeffs.template cast<T>();
    }

    auto get_tmp3()
    {
        return coeffs.template head<3>();
    }

    // Helper to get M_l^m as complex
    std::complex<T> get_coeff(int l, int m) const
    {
        int absm = std::abs(m);
        int base = l * l;
        T re, im = 0;
        if (m == 0)
        {
            re = coeffs[base];
        }
        else
        {
            re = coeffs[base + 2 * absm - 1];
            im = coeffs[base + 2 * absm];
        }
        if (m < 0)
        {
            int sign = ((absm % 2 == 0) ? 1 : -1);
            // T temp = re;
            re = sign * re;
            im = -sign * im;
        }
        return { re, im };
    }

    // Helper to set M_l^m from complex
    void set_coeff(int l, int m, std::complex<T> val)
    {
        int absm = std::abs(m);
        if (m < 0)
        {
            float sign = ((absm % 2 == 0) ? 1 : -1);
            val = complex<T>{ sign, 0.f } * conj(val);
            m = absm;
        }
        int base = l * l;
        if (m == 0)
        {
            coeffs[base] = val.real();
        }
        else
        {
            coeffs[base + 2 * m - 1] = val.real();
            coeffs[base + 2 * m] = val.imag();
        }
    }

    // Helper functions for general Ylm
    static double factorial(int n)
    {
        if (n <= 1)
            return 1;
        double res = 1;
        for (int i = 2; i <= n; ++i)
            res *= i;
        return res;
    }

    static double double_factorial(int n)
    {
        if (n <= 0)
            return 1;
        double res = n;
        for (int i = n - 2; i > 0; i -= 2)
            res *= i;
        return res;
    }

    static T assoc_legendre(int l, int m, T x)
    {
        m = std::abs(m);
        if (m > l)
            return T(0);
        T sign = (m % 2 == 0 ? T(1) : T(-1));
        T df = double_factorial(2 * m - 1);
        T omx2 = T(1) - x * x;
        T pmm = sign * df * pow(omx2, T(m) / T(2));
        if (l == m)
            return pmm;
        T pmmp1 = x * T(2 * m + 1) * pmm;
        if (l == m + 1)
            return pmmp1;
        T old_pmm = pmm;
        T old_pmmp1 = pmmp1;
        for (int ll = m + 2; ll <= l; ++ll)
        {
            T pll = (x * T(2 * ll - 1) * old_pmmp1 - T(ll + m - 1) * old_pmm) / T(ll - m);
            old_pmm = old_pmmp1;
            old_pmmp1 = pll;
        }
        return old_pmmp1;
    }

    // General spherical harmonics Y_l^m (theta, phi) with Condon-Shortley phase
    std::complex<T> Ylm(int l, int m, T theta, T phi) const
    {
        if (m < 0)
            return pow(T(-1), -m) * conj(Ylm(l, -m, theta, phi));
        T x = cos(theta);
        T p = assoc_legendre(l, m, x);
        // T pi_val = acos(T(-1));
        double norm = sqrt(double(2 * l + 1) / (4 * PI) * factorial(l - m) / factorial(l + m));
        complex<T> eip = exp(complex<T>(T(0), T(m) * phi));
        return static_cast<T>(norm) * p * eip;
    }

    static multipole from_particle(Vector<T, 3> a, T mass, uint_type ID = 0)
    {
        (void)ID;
        a = -a;
        multipole M;
        T r = a.norm();

        // gpu_assert(r != T(0));
        /*
          if (r == T(0)) {
                M.set_coeff(0, 0, complex<T>(-mass * sqrt(T(4) * acos(T(-1))), T(0)));
                return M;
            }
        */
        T theta = acos(a[2] / r);
        T phi = atan2(a[1], a[0]);
        // T pi = acos(T(-1));
        for (int l = 0; l <= (int)N; ++l)
        {
            for (int m = -l; m <= l; ++m)
            {
                complex<T> ylm = M.Ylm(l, m, theta, phi); // Use M to call Ylm since it's const
                complex<T> val = (-mass) * pow(r, T(l)) * static_cast<cpu_T>(4 * PI / double(2 * l + 1)) * conj(ylm);
                M.set_coeff(l, m, val);
            }
        }
        return M;
    }

    multipole shift_ext(Vector<T, 3> a) const
    {
        multipole ret;
        T r = a.norm();
        T theta = acos(a[2] / r);
        T phi = atan2(a[1], a[0]);
        for (int l = 0; l <= (int)N; ++l)
        {
            for (int m = -l; m <= l; ++m)
            {
                complex<T> sum = { 0, 0 };
                for (int j = 0; j <= l; ++j)
                {
                    for (int k = -j; k <= j; ++k)
                    {
                        sum += get_coeff(j, k) * pow(r, l - j) * Ylm(l - j, m - k, theta, phi);
                    }
                }
                ret.set_coeff(l, m, sum);
            }
        }
        return ret;
    }

    multipole shift_loc(Vector<T, 3> a) const
    {
        a = -a;
        T r = a.norm();
        T theta = acos(a[2] / r);
        T phi = atan2(a[1], a[0]);
        multipole ret;
        for (int j = 0; j <= (int)N; ++j)
        {
            for (int k = -j; k <= j; ++k)
            {
                complex<T> sum = { 0, 0 };
                for (int l = j; l <= (int)N; ++l)
                {
                    for (int s = -l; s <= l; ++s)
                    {
                        sum += get_coeff(l, s) * pow(r, l - j) * Ylm(l - j, k - s, theta, phi);
                    }
                }
                ret.set_coeff(j, k, sum);
            }
        }
        return ret;
    }

    multipole makelocal(Vector<T, 3> a) const
    {
        a = -a;
        T r = a.norm();
        T theta = acos(a[2] / r);
        T phi = atan2(a[1], a[0]);
        multipole ret;
        for (int j = 0; j <= (int)N; ++j)
        {
            for (int k = -j; k <= j; ++k)
            {
                complex<T> sum = { 0, 0 };
                for (int n = 0; n <= (int)N; ++n)
                {
                    for (int m = -n; m <= n; ++m)
                    {
                        sum += get_coeff(n, m) * pow(T(-1), n) * Ylm(j + n, m - k, theta, phi) / pow(r, j + n + 1);
                    }
                }
                ret.set_coeff(j, k, sum);
            }
        }
        return ret;
    }

    // Wigner small d-matrix d^j_{m mp}(beta)
    double d_wigner(int j, int m, int mp, double beta) const
    {
        double cb2 = std::cos(beta / double(2));
        double sb2 = std::sin(beta / double(2));
        double sqrt_fact = std::sqrt(factorial(j + m) * factorial(j - m) * factorial(j + mp) * factorial(j - mp));
        double sum = double(0);
        for (int k = 0; k <= 2 * j; ++k)
        {
            int den1 = j - mp - k;
            int den2 = j + m - k;
            int den3 = k;
            int den4 = k + mp - m;
            if (den1 < 0 || den2 < 0 || den3 < 0 || den4 < 0)
                continue;
            double denom = factorial(den1) * factorial(den2) * factorial(den3) * factorial(den4);
            if (denom == double(0))
                continue;
            double pow_cb = std::pow(cb2, 2 * j + m - mp - 2 * k);
            double pow_sb = std::pow(sb2, 2 * k + mp - m);
            double term = std::pow(double(-1), k) * pow_cb * pow_sb / denom;
            sum += term;
        }
        return sqrt_fact * sum;
    }

    multipole rot(Tint step = 1) const
    {
        if (step < 0)
            return rot(step + 3);
        if (step == 0)
            return *this;
        multipole ret;
        // const T pi = std::acos(T(-1));
        const double alpha = PI;
        const double beta = PI / 2;
        const double gamma = PI / 2;
        for (int l = 0; l <= (int)N; ++l)
        {
            for (int m = -l; m <= l; ++m)
            {
                std::complex<T> sum = { 0, 0 };
                for (int k = -l; k <= l; ++k)
                {
                    double d = d_wigner(l, m, k, beta);
                    std::complex<double> D = std::exp(std::complex<double>(0, -m * alpha)) * d
                                             * std::exp(std::complex<double>(0, -k * gamma));
                    sum += cast<T>(D) * get_coeff(l, k);
                }
                ret.set_coeff(l, m, sum);
            }
        }
        return ret.rot(step - 1);
    }

#if CALC_POTENTIAL
    T calc_loc_potential(Vector<T, 3> r) const
    {
        r = -r;
        T ret = 0;
        T rho = r.norm();
        T theta = acos(r[2] / rho);
        T phi = atan2(r[1], r[0]);
        for (int j = 0; j <= (int)N; ++j)
        {
            for (int k = -j; k <= j; ++k)
            {
                ret += (get_coeff(j, k) * pow(rho, j) * Ylm(j, k, theta, phi)).real();
            }
        }
        return ret;
    }
#endif

    Vector<T, 3> calc_force(Vector<T, 3> r) const
    {
        r = -r;
        T rho = r.norm();
        T theta = acos(r[2] / rho);
        T phi = atan2(r[1], r[0]);
        // Compute -grad (sum L_j^k rho^j Y_j^k)
        // Use spherical grad formula: grad f = df/dr hat r + (1/r) df/dtheta hat theta + (1/(r sin theta)) df/dphi hat
        // phi Compute partial derivatives numerically with small epsilon for simplicity (since N small)
        T eps = 1e-8f;
        Vector<T, 3> F;
        for (int i = 0; i < 3; ++i)
        {
            Vector<T, 3> rp = r;
            rp[i] += eps;
            T phip = calc_loc_potential(rp);
            Vector<T, 3> rm = r;
            rm[i] -= eps;
            T phim = calc_loc_potential(rm);
            F[i] = -(phip - phim) / (2 * eps);
        }
        return F;
    }

    template<class STREAM>
    friend STREAM& operator<<(STREAM& s, const multipole& mm)
    {
        s << "{ ";
        for (int l = 0; l <= (int)N; ++l)
        {
            for (int m = -l; m <= l; ++m)
            {
                s << mm.get_coeff(l, m) << " ";
            }
            s << " | ";
        }
        s << "}";
        return s;
    }
};
