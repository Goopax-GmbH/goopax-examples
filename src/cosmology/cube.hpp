#include <Eigen/Dense>

// Gravitational potential and field of a homogeneous (solid) cube, evaluated by
// inclusion-exclusion over the 8 corners of the box relative to the field point.
//
// These are the standard "Nagy" prism formulas. For the antiderivative
//   F = yz ln(x+r) + xz ln(y+r) + xy ln(z+r)
//       - x^2/2 atan(yz/(xr)) - y^2/2 atan(xz/(yr)) - z^2/2 atan(xy/(zr))
// (with r = sqrt(x^2+y^2+z^2)) the field components are the closed-form
//   dF/dx = y ln(z+r) + z ln(y+r) - x atan(yz/(xr))
//   dF/dy = z ln(x+r) + x ln(z+r) - y atan(xz/(yr))
//   dF/dz = x ln(y+r) + y ln(x+r) - z atan(xy/(zr))
// Terms that vanish under the alternating 8-corner sum are intentionally
// dropped, which keeps every contribution bounded: each large atan argument is
// multiplied by exactly the small coordinate that produces it.

// Numerically stable log(c + r).
// Since r >= |c|, the argument c+r is always >= 0, but for c < 0 the direct
// evaluation suffers catastrophic cancellation. We then use the algebraically
// equivalent (c+r) = perp2 / (r-c), where perp2 = r^2 - c^2 is the sum of the
// squares of the other two coordinates and r-c is free of cancellation.
template<typename T>
T stable_log_sum(T c, T r, T perp2, T eps)
{
    T arg = cond(c >= T(0), c + r, perp2 / (r - c + eps));
    return log(arg + eps);
}

template<typename T>
struct cube_terms
{
    T phi;
    T fx;
    T fy;
    T fz;
};

// Potential antiderivative and its gradient for a single corner offset (x,y,z).
// All transcendentals are computed once and shared between the potential and
// the three force components.
template<typename T>
cube_terms<T> cube_primitive_and_grad(T x, T y, T z)
{
    const T eps = T(1e-30f);

    T x2 = x * x;
    T y2 = y * y;
    T z2 = z * z;
    T r = sqrt(x2 + y2 + z2 + eps);

    T log_x = stable_log_sum(x, r, y2 + z2, eps);
    T log_y = stable_log_sum(y, r, x2 + z2, eps);
    T log_z = stable_log_sum(z, r, x2 + y2, eps);

    // atan(prod / (coord * r)); the additive eps only guards the exact 0/0 case
    // and is negligible otherwise (the result is multiplied by `coord`, which
    // vanishes in precisely the regime where the denominator does).
    T atan_x = atan(y * z / (x * r + eps));
    T atan_y = atan(x * z / (y * r + eps));
    T atan_z = atan(x * y / (z * r + eps));

    cube_terms<T> t;
    t.phi = y * z * log_x + x * z * log_y + x * y * log_z
            - T(0.5f) * (x2 * atan_x + y2 * atan_y + z2 * atan_z);
    t.fx = y * log_z + z * log_y - x * atan_x;
    t.fy = z * log_x + x * log_z - y * atan_y;
    t.fz = x * log_y + y * log_x - z * atan_z;
    return t;
}

template<typename T>
pair<T, Eigen::Vector<T, 3>> get_potential_and_force(const Eigen::Vector<T, 3>& pos, T a, T rho)
{
    T half = a / 2.f;
    T low_x = -half - pos[0];
    T low_y = -half - pos[1];
    T low_z = -half - pos[2];
    T high_x = half - pos[0];
    T high_y = half - pos[1];
    T high_z = half - pos[2];

    T sum_phi = 0.0;
    T sum_fx = 0.0;
    T sum_fy = 0.0;
    T sum_fz = 0.0;

    for (int ix = 0; ix < 2; ++ix)
    {
        T xx = (ix == 0 ? low_x : high_x);
        for (int iy = 0; iy < 2; ++iy)
        {
            T yy = (iy == 0 ? low_y : high_y);
            for (int iz = 0; iz < 2; ++iz)
            {
                T zz = (iz == 0 ? low_z : high_z);
                int num_low = (ix == 0) + (iy == 0) + (iz == 0);
                T sign = (num_low % 2 == 0 ? 1.0 : -1.0);
                cube_terms<T> t = cube_primitive_and_grad(xx, yy, zz);
                sum_phi += sign * t.phi;
                sum_fx += sign * t.fx;
                sum_fy += sign * t.fy;
                sum_fz += sign * t.fz;
            }
        }
    }

    T phi = -rho * sum_phi;
    Eigen::Vector<T, 3> force(-rho * sum_fx, -rho * sum_fy, -rho * sum_fz);

    return { phi, force };
}
