#pragma once
#include "zero.hpp"

template<unsigned int N>
vector<vector<Tuint>> make_indices()
{
    vector<vector<Tuint>> sub = make_indices<N - 1>();
    vector<vector<Tuint>> ret;
    for (auto s : sub)
    {
        for (Tuint k = 0; k < 3; ++k)
        {
            ret.push_back(s);
            ret.back().push_back(k);
        }
    }
    return ret;
}
template<>
vector<vector<Tuint>> make_indices<0>()
{
    return { {} };
}

template<unsigned int N>
vector<Tuint> make_mi()
{
    std::map<vector<Tuint>, Tuint> have;
    vector<Tuint> ret;
    Tuint pos = 0;
    auto indices = make_indices<N>();
    for (auto i : indices)
    {
        vector<Tuint> s = i;
        sort(s.begin(), s.end());
        if (have.find(s) == have.end())
            have[s] = pos++;
        ret.push_back(have[s]);
    }
    return ret;
}

const auto MI2 = reinterpret<Vector<Vector<Tuint, 3>, 3>>(make_mi<2>());
const auto MI3 = reinterpret<Vector<Vector<Vector<Tuint, 3>, 3>, 3>>(make_mi<3>());
const auto MI4 = reinterpret<Vector<Vector<Vector<Vector<Tuint, 3>, 3>, 3>, 3>>(make_mi<4>());

template<class T, unsigned int N>
struct multipole
{
    using goopax_struct_type = T;
    template<typename X>
    using goopax_struct_changetype = multipole<typename goopax_struct_changetype<T, X>::type, N>;
    using uint_type = typename change_gpu_mode<unsigned int, T>::type;

    using Tbf16 = typename change_gpu_mode<bfloat16_t, T>::type;

    using T1 = T;
    using T2 = T;
    using T3 = T;
    using T4 = T;

    T A;
    Vector<T1, 3 * (N >= 1)> B;
    Vector<T2, 6 * (N >= 2)> C;
    Vector<T3, 10 * (N >= 3)> D;
    Vector<T4, 15 * (N >= 4)> E;

    multipole rot(Tint step = 1) const
    {
        if (step < 0)
            return rot(step + 3);
        if (step == 0)
            return *this;
        multipole ret;
        ret.A = A;
        for (Tuint i = 0; i < 3; ++i)
        {
            Tuint io = (i + 1) % 3;
            if (N >= 1)
                ret.B[i] = B[io];
            for (Tuint k = 0; k < 3; ++k)
            {
                Tuint ko = (k + 1) % 3;
                if (N >= 2)
                    ret.C[MI2[i][k]] = C[MI2[io][ko]];
                for (Tuint l = 0; l < 3; ++l)
                {
                    Tuint lo = (l + 1) % 3;
                    if (N >= 3)
                        ret.D[MI3[i][k][l]] = D[MI3[io][ko][lo]];
                    for (Tuint m = 0; m < 3; ++m)
                    {
                        Tuint mo = (m + 1) % 3;
                        if (N >= 4)
                            ret.E[MI4[i][k][l][m]] = E[MI4[io][ko][lo][mo]];
                    }
                }
            }
        }
        return ret.rot(step - 1);
    }

    template<class U>
    multipole(const multipole<U, N>& b)
        : A(b.A)
    {
        std::copy(b.B.begin(), b.B.end(), B.begin());
        std::copy(b.C.begin(), b.C.end(), C.begin());
        std::copy(b.D.begin(), b.D.end(), D.begin());
        std::copy(b.E.begin(), b.E.end(), E.begin());
    }

    multipole()
    {
    }

    multipole& operator+=(const multipole& b)
    {
        A += b.A;
        if (N >= 1)
            B += b.B;
        if (N >= 2)
            C += b.C;
        if (N >= 3)
            D += b.D;
        if (N >= 4)
            E += b.E;
        return *this;
    }

    static multipole from_particle(Vector<T, 3> aa, T mass)
    {
        aa = -aa;
        multipole M;
        if (N >= 0)
        {
            M.A = -mass;
        }
        if (N >= 1)
        {
            const Vector<T1, 3> a = aa.template cast<T1>();

            for (Tuint k = 0; k < 3; ++k)
            {
                M.B[k] = (-mass) * a[k];
            }
        }
        if (N >= 2)
        {
            const Vector<T2, 3> a = aa.template cast<T2>();

            for (Tuint i = 0; i < 3; ++i)
                for (Tuint k = i; k < 3; ++k)
                {
                    M.C[MI2[i][k]] = (-mass) * (1.5f * a[i] * a[k] - 0.5f * int(i == k) * a.squaredNorm());
                }
        }
        if (N >= 3)
        {
            const Vector<T3, 3> a = aa.template cast<T3>();

            for (Tuint i = 0; i < 3; ++i)
                for (Tuint k = i; k < 3; ++k)
                    for (Tuint l = k; l < 3; ++l)
                    {
                        M.D[MI3[i][k][l]] =
                            (-mass)
                            * (2.5f * a[i] * a[k] * a[l]
                               - 0.5f * a.squaredNorm()
                                     * (a[i] * Tint(k == l) + a[k] * Tint(i == l) + a[l] * Tint(i == k)));
                    }
        }
        if (N >= 4)
        {
            const Vector<T4, 3> a = aa.template cast<T4>();

            for (Tuint i = 0; i < 3; ++i)
                for (Tuint k = i; k < 3; ++k)
                    for (Tuint l = k; l < 3; ++l)
                        for (Tuint m = l; m < 3; ++m)
                        {
                            M.E[MI4[i][k][l][m]] =
                                static_cast<T4>(-mass)
                                * (static_cast<T4>(35.0f / 8) * a[i] * a[k] * a[l] * a[m]
                                   - static_cast<T4>(5.0f / 8)
                                         * (a[i] * a[k] * (l == m) + a[i] * a[l] * (k == m) + a[i] * a[m] * (k == l)
                                            + a[k] * a[l] * (i == m) + a[k] * a[m] * (i == l) + a[l] * a[m] * (i == k))
                                         * a.squaredNorm()
                                   + static_cast<T4>(1.0f / 8) * pow2(a.squaredNorm())
                                         * ((i == k) * (l == m) + (i == l) * (k == m) + (i == m) * (k == l)));
                        }
        }
        return M;
    }

    static multipole zero()
    {
        return from_particle({ 0, 0, 0 }, 0);
    }

    multipole scale_ext(T s) const
    {
        multipole ret = *this;

        if (N >= 1)
        {
            ret.B *= s;
        }
        if (N >= 2)
        {
            ret.C *= pow2(s);
        }
        if (N >= 3)
        {
            ret.D *= pow3(s);
        }
        if (N >= 4)
        {
            ret.E *= pow4(s);
        }
        return ret;
    }

    multipole scale_loc(T s) const
    {
        T inv_s = 1.f / s;
        multipole ret = *this;

        ret.A *= inv_s;
        if (N >= 1)
        {
            ret.B *= pow2(inv_s);
        }
        if (N >= 2)
        {
            ret.C *= pow3(inv_s);
        }
        if (N >= 3)
        {
            ret.D *= pow4(inv_s);
        }
        if (N >= 4)
        {
            ret.E *= pow5(inv_s);
        }
        return ret;
    }

    multipole shift_ext(Vector<T, 3> aa) const
    {
        multipole M = *this;

        if (N >= 1)
        {
            const Vector<T1, 3> a = aa.template cast<T1>();
            M.B += a * A;
        }
        if (N >= 2)
        {
            const Vector<T2, 3> a = aa.template cast<T2>();

            for (Tuint i = 0; i < 3; ++i)
                for (Tuint k = i; k < 3; ++k)
                {
                    M.C[MI2[i][k]] += 1.5f * a[i] * a[k] * A - 0.5f * A * a.squaredNorm() * Tint(i == k)
                                      + 1.5f * (B[i] * a[k] + B[k] * a[i]);
                    for (Tuint n = 0; n < 3; ++n)
                    {
                        M.C[MI2[i][k]] += -B[n] * a[n] * Tint(i == k);
                    }
                }
        }
        if (N >= 3)
        {
            const Vector<T3, 3> a = aa.template cast<T3>();

            for (Tuint i = 0; i < 3; ++i)
                for (Tuint k = i; k < 3; ++k)
                    for (Tuint l = k; l < 3; ++l)
                    {
                        M.D[MI3[i][k][l]] += 2.5f * A * a[i] * a[k] * a[l]
                                             - 0.5f * A * a.squaredNorm()
                                                   * (a[i] * Tint(k == l) + a[k] * Tint(i == l) + a[l] * Tint(i == k))
                                             + static_cast<T>(5.0 / 3)
                                                   * (C[MI2[i][k]] * a[l] + C[MI2[i][l]] * a[k] + C[MI2[k][l]] * a[i])
                                             + 5.0f / 2 * (B[i] * a[k] * a[l] + B[k] * a[i] * a[l] + B[l] * a[i] * a[k])
                                             - 1.0f / 2 * a.squaredNorm()
                                                   * (B[i] * Tint(k == l) + B[k] * Tint(i == l) + B[l] * Tint(i == k));
                        for (Tuint n = 0; n < 3; ++n)
                        {
                            M.D[MI3[i][k][l]] +=
                                -static_cast<T>(2.0 / 3) * a[n]
                                    * (C[MI2[n][k]] * Tint(i == l) + C[MI2[n][i]] * Tint(k == l)
                                       + C[MI2[n][l]] * Tint(i == k))
                                - a[n] * B[n] * (a[i] * Tint(k == l) + a[k] * Tint(i == l) + a[l] * Tint(i == k));
                        }
                    }
        }
        if (N >= 4)
        {
            const Vector<T4, 3> a = aa.template cast<T4>();
            const T4 A = static_cast<T4>(this->A);
            const Vector<T4, 3 * (N >= 1)> B = this->B.template cast<T4>();
            const Vector<T4, 6 * (N >= 2)> C = this->C.template cast<T4>();
            const Vector<T4, 10 * (N >= 3)> D = this->D.template cast<T4>();

            for (Tuint i = 0; i < 3; ++i)
                for (Tuint k = i; k < 3; ++k)
                    for (Tuint l = k; l < 3; ++l)
                        for (Tuint m = l; m < 3; ++m)
                        {
                            M.E[MI4[i][k][l][m]] +=
                                static_cast<T4>(35.0f / 8) * A * a[i] * a[k] * a[l] * a[m]
                                - static_cast<T4>(5.0f / 8) * A * a.squaredNorm()
                                      * (a[i] * a[k] * (l == m) + a[i] * a[l] * (k == m) + a[i] * a[m] * (k == l)
                                         + a[k] * a[l] * (i == m) + a[k] * a[m] * (i == l) + a[l] * a[m] * (i == k))
                                + static_cast<T4>(1.0f / 8) * A * pow2(a.squaredNorm())
                                      * ((i == k) * (l == m) + (i == l) * (k == m) + (i == m) * (k == l))
                                + static_cast<T4>(7.0f / 4)
                                      * (D[MI3[i][k][l]] * a[m] + D[MI3[i][k][m]] * a[l] + D[MI3[i][l][m]] * a[k]
                                         + D[MI3[k][l][m]] * a[i])
                                + static_cast<T4>(35.0 / 12)
                                      * (C[MI2[i][k]] * a[l] * a[m] + C[MI2[i][l]] * a[k] * a[m]
                                         + C[MI2[i][m]] * a[k] * a[l] + C[MI2[k][l]] * a[i] * a[m]
                                         + C[MI2[k][m]] * a[i] * a[l] + C[MI2[l][m]] * a[i] * a[k])
                                - static_cast<T4>(5.0 / 12) * a.squaredNorm()
                                      * (C[MI2[i][k]] * (l == m) + C[MI2[i][l]] * (k == m) + C[MI2[i][m]] * (k == l)
                                         + C[MI2[k][l]] * (i == m) + C[MI2[k][m]] * (i == l) + C[MI2[l][m]] * (i == k))
                                + static_cast<T4>(35.0 / 8)
                                      * (B[i] * a[k] * a[l] * a[m] + B[k] * a[i] * a[l] * a[m]
                                         + B[l] * a[i] * a[k] * a[m] + B[m] * a[i] * a[k] * a[l])
                                - static_cast<T4>(5.0f / 8) * a.squaredNorm()
                                      * (B[i] * (a[k] * (l == m) + a[l] * (k == m) + a[m] * (k == l))
                                         + B[k] * (a[i] * (l == m) + a[l] * (i == m) + a[m] * (i == l))
                                         + B[l] * (a[i] * (k == m) + a[k] * (i == m) + a[m] * (i == k))
                                         + B[m] * (a[i] * (k == l) + a[k] * (i == l) + a[l] * (i == k)));
                            for (Tuint n = 0; n < 3; ++n)
                            {
                                M.E[MI4[i][k][l][m]] +=
                                    -static_cast<T4>(0.5f) * a[n]
                                        * (D[MI3[n][i][k]] * Tint(l == m) + D[MI3[n][i][l]] * Tint(k == m)
                                           + D[MI3[n][i][m]] * Tint(k == l) + D[MI3[n][k][l]] * Tint(i == m)
                                           + D[MI3[n][k][m]] * Tint(i == l) + D[MI3[n][l][m]] * Tint(i == k))
                                    - static_cast<T4>(5.0 / 6) * a[n]
                                          * (C[MI2[n][i]]
                                                 * (a[k] * Tint(l == m) + a[l] * Tint(k == m) + a[m] * Tint(k == l))
                                             + C[MI2[n][k]]
                                                   * (a[i] * Tint(l == m) + a[l] * Tint(i == m) + a[m] * Tint(i == l))
                                             + C[MI2[n][l]]
                                                   * (a[i] * Tint(k == m) + a[k] * Tint(i == m) + a[m] * Tint(i == k))
                                             + C[MI2[n][m]]
                                                   * (a[i] * Tint(k == l) + a[k] * Tint(i == l) + a[l] * Tint(i == k)))
                                    - static_cast<T4>(5.0f / 4) * B[n] * a[n]
                                          * (a[i] * a[k] * Tint(l == m) + a[i] * a[l] * Tint(k == m)
                                             + a[i] * a[m] * Tint(k == l) + a[k] * a[l] * Tint(i == m)
                                             + a[k] * a[m] * Tint(i == l) + a[l] * a[m] * Tint(i == k))
                                    + static_cast<T4>(0.5f) * a.squaredNorm() * a[n] * B[n]
                                          * (Tint(i == k) * Tint(l == m) + Tint(i == l) * Tint(k == m)
                                             + Tint(i == m) * Tint(k == l));
                                for (Tuint o = 0; o < 3; ++o)
                                {
                                    M.E[MI4[i][k][l][m]] += static_cast<T4>(1.0 / 3) * C[MI2[n][o]] * a[n] * a[o]
                                                            * (Tint(i == k) * Tint(l == m) + Tint(i == l) * Tint(k == m)
                                                               + Tint(i == m) * Tint(k == l));
                                }
                            }
                        }
        }
        return M;
    }

    multipole shift_loc(Vector<T, 3> a) const
    {
        a = -a;
        multipole M = *this;
        if (N >= 1)
        {
            for (Tuint i = 0; i < 3; ++i)
            {
                M.A += B[i] * a[i];
            }
        }
        if (N >= 2)
        {
            for (Tuint i = 0; i < 3; ++i)
                for (Tuint k = 0; k < 3; ++k)
                {
                    M.A += C[MI2[i][k]] * a[i] * a[k];
                    M.B[i] += 2 * C[MI2[i][k]] * a[k];
                }
        }
        if (N >= 3)
        {
            for (Tuint i = 0; i < 3; ++i)
                for (Tuint k = 0; k < 3; ++k)
                    for (Tuint l = 0; l < 3; ++l)
                    {
                        M.A += D[MI3[i][k][l]] * a[i] * a[k] * a[l];
                        M.B[i] += 3 * D[MI3[i][k][l]] * a[k] * a[l];
                        if (i <= k)
                            M.C[MI2[i][k]] += 3 * D[MI3[i][k][l]] * a[l];
                    }
        }
        if (N >= 4)
        {
            for (Tuint i = 0; i < 3; ++i)
                for (Tuint k = 0; k < 3; ++k)
                    for (Tuint l = 0; l < 3; ++l)
                        for (Tuint m = 0; m < 3; ++m)
                        {
                            M.A += E[MI4[i][k][l][m]] * a[i] * a[k] * a[l] * a[m];
                            M.B[i] += 4 * E[MI4[i][k][l][m]] * a[k] * a[l] * a[m];
                            if (i <= k)
                                M.C[MI2[i][k]] += 6 * E[MI4[i][k][l][m]] * a[l] * a[m];
                            if (i <= k && k <= l)
                                M.D[MI3[i][k][l]] += 4 * E[MI4[i][k][l][m]] * a[m];
                        }
        }
        return M;
    }

    multipole makelocal(Vector<T, 3> a) const
    {
        a = -a;
        multipole M;
        T inva = pow<-1, 2>(a.squaredNorm());
        const Vector<T1, 3> e = a * inva;

        M.A = inva * A;
        if (N >= 1)
        {
            for (Tuint n = 0; n < 3; ++n)
            {
                M.B[n] = -pow2(inva) * A * e[n];
                M.A += pow2(inva) * B[n] * e[n];
            }
        }
        if (N >= 2)
        {
            for (Tuint i = 0; i < 3; ++i)
            {
                for (Tuint k = 0; k < 3; ++k)
                {
                    if (i <= k)
                        M.C[MI2[i][k]] = pow3(inva) * (1.5f * A * e[i] * e[k] - 0.5f * A * Tint(i == k));
                    M.B[i] += pow3(inva) * (-3 * B[k] * e[k] * e[i]);
                    M.A += pow3(inva) * C[MI2[i][k]] * e[i] * e[k];
                }
            }
            M.B += pow3(inva) * B;
        }
        if (N >= 3)
        {
            for (Tuint i = 0; i < 3; ++i)
            {
                for (Tuint k = 0; k < 3; ++k)
                {
                    for (Tuint l = 0; l < 3; ++l)
                    {
                        if (i <= k && k <= l)
                            M.D[MI3[i][k][l]] =
                                pow4(inva)
                                * (-5.0f / 2 * A * e[i] * e[k] * e[l]
                                   + 0.5f * A * (e[i] * Tint(k == l) + e[k] * Tint(i == l) + e[l] * Tint(i == k)));
                        if (i <= k)
                            M.C[MI2[i][k]] +=
                                pow4(inva)
                                * (15.0f / 2 * B[l] * e[l] * e[i] * e[k] - 1.5f * B[l] * e[l] * Tint(i == k));
                        M.B[i] += pow4(inva) * (-5 * C[MI2[k][l]] * e[k] * e[l] * e[i]);
                        M.A += pow4(inva) * (D[MI3[i][k][l]] * e[i] * e[k] * e[l]);
                    }
                    if (i <= k)
                        M.C[MI2[i][k]] += pow4(inva) * (-1.5f * (B[i] * e[k] + B[k] * e[i]));
                    M.B[i] += pow4(inva) * 2 * C[MI2[i][k]] * e[k];
                }
            }
        }
        if (N >= 4)
        {
            for (Tuint i = 0; i < 3; ++i)
            {
                for (Tuint k = 0; k < 3; ++k)
                {
                    for (Tuint l = 0; l < 3; ++l)
                    {
                        for (Tuint m = 0; m < 3; ++m)
                        {
                            if (i <= k && k <= l && l <= m)
                            {
                                const Vector<T4, 3> e = (a * inva).template cast<T4>();
                                const T4 A = static_cast<T4>(this->A);

                                M.E[MI4[i][k][l][m]] =
                                    static_cast<T4>(pow5(inva))
                                    * (static_cast<T4>(35.0f / 8) * A * e[i] * e[k] * e[l] * e[m]
                                       + static_cast<T4>(1.0f / 8) * A
                                             * ((i == k) * (l == m) + (i == l) * (k == m) + (i == m) * (k == l))
                                       - static_cast<T4>(5.0f / 8) * A
                                             * (e[i] * e[k] * Tint(l == m) + e[i] * e[l] * Tint(k == m)
                                                + e[i] * e[m] * Tint(k == l) + e[k] * e[l] * Tint(i == m)
                                                + e[k] * e[m] * Tint(i == l) + e[l] * e[m] * Tint(i == k)));
                            }
                            if (i <= k && k <= l)
                                M.D[MI3[i][k][l]] +=
                                    pow5(inva)
                                    * (-35.0f / 2 * B[m] * e[m] * e[i] * e[k] * e[l]
                                       + 5.0f / 2 * B[m] * e[m]
                                             * (e[i] * Tint(k == l) + e[k] * Tint(i == l) + e[l] * Tint(i == k)));
                            if (i <= k)
                                M.C[MI2[i][k]] += pow5(inva)
                                                  * (35.0f / 2 * C[MI2[l][m]] * e[l] * e[m] * e[i] * e[k]
                                                     - 5.0f / 2 * C[MI2[l][m]] * e[l] * e[m] * Tint(i == k));
                            M.B[i] += pow5(inva) * (-7 * D[MI3[k][l][m]] * e[k] * e[l] * e[m] * e[i]);
                            M.A += pow5(inva) * (E[MI4[i][k][l][m]] * e[i] * e[k] * e[l] * e[m]);
                        }
                        if (i <= k && k <= l)
                            M.D[MI3[i][k][l]] +=
                                pow5(inva)
                                * (+5.0f / 2 * (B[i] * e[k] * e[l] + B[k] * e[i] * e[l] + B[l] * e[i] * e[k])
                                   - 0.5f * (B[i] * Tint(k == l) + B[k] * Tint(i == l) + B[l] * Tint(i == k)));
                        if (i <= k)
                            M.C[MI2[i][k]] += pow5(inva) * (-5 * e[l] * (C[MI2[l][k]] * e[i] + C[MI2[l][i]] * e[k]));
                        M.B[i] += pow5(inva) * 3 * D[MI3[i][k][l]] * e[k] * e[l];
                    }
                    if (i <= k)
                        M.C[MI2[i][k]] += pow5(inva) * C[MI2[i][k]];
                }
            }
        }
        return M;
    }

#if CALC_POTENTIAL
    T calc_loc_potential(Vector<T, 3> r) const
    {
        r = -r;
        T ret = A;
        for (Tuint i = 0; i < 3; ++i)
        {
            if (N >= 1)
                ret += B[i] * r[i];
            for (Tuint k = i; k < 3; ++k)
            {
                if (N >= 2)
                    ret += C[MI2[i][k]] * r[i] * r[k] * (i == k ? 1 : 2);
                for (Tuint l = k; l < 3; ++l)
                {
                    if (N >= 3)
                        ret += D[MI3[i][k][l]] * r[i] * r[k] * r[l] * (i == l ? 1 : (i == k || k == l ? 3 : 6));
                    for (Tuint m = l; m < 3; ++m)
                    {
                        if (N >= 4)
                            ret += E[MI4[i][k][l][m]] * r[i] * r[k] * r[l] * r[m]
                                   * (i == m ? 1
                                             : (i == l || k == m
                                                    ? 4
                                                    : (i == k && l == m ? 6 : (i == k || k == l || l == m ? 12 : 24))));
                    }
                }
            }
        }
        return ret;
    }
#endif

    Vector<T, 3> calc_force(Vector<T, 3> r) const
    {
        r = -r;
        Vector<T, 3> F = { 0, 0, 0 };
        for (Tuint i = 0; i < 3; ++i)
        {
            if (N >= 1)
                F[i] += B[i];
            for (Tuint k = 0; k < 3; ++k)
            {
                if (N >= 2)
                    F[i] += 2 * (C[MI2[k][i]]) * r[k];

                for (Tuint l = 0; l < 3; ++l)
                {
                    if (N >= 3)
                        F[i] += 3 * D[MI3[i][k][l]] * r[k] * r[l];
                    for (Tuint m = 0; m < 3; ++m)
                    {
                        if (N >= 4)
                            F[i] += 4 * E[MI4[i][k][l][m]] * r[k] * r[l] * r[m];
                    }
                }
            }
        }
        return F;
    }
    template<class STREAM>
    friend STREAM& operator<<(STREAM& s, const multipole& m)
    {
        s << "{A=" << m.A;
        if (N >= 1)
            s << ", B=" << m.B;
        if (N >= 2)
            s << ", C=" << m.C;
        if (N >= 3)
            s << ", D=" << m.D;
        if (N >= 4)
            s << ", E=" << m.E;
        s << "}";
        return s;
    }
};
