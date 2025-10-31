#include <Eigen/Dense>

template<typename T>
T primitive(T x, T y, T z)
{
    T r2 = x * x + y * y + z * z;
    T r = sqrt(r2 + 1e-30f);
    T log_xr = log(abs(x + r) + 1e-30f);
    T log_yr = log(abs(y + r) + 1e-30f);
    T log_zr = log(abs(z + r) + 1e-30f);
    T atan1 = atan(y * z / (x * r + 1e-30f));
    T atan2 = atan(x * z / (y * r + 1e-30f));
    T atan3 = atan(x * y / (z * r + 1e-30f));
    return y * z * log_xr - 0.5f * x * x * atan1 + x * z * log_yr - 0.5f * y * y * atan2 + x * y * log_zr
           - 0.5f * z * z * atan3;
}

template<typename T>
T partial_dx(T x, T y, T z)
{
    T r2 = x * x + y * y + z * z;
    T r = sqrt(r2 + 1e-30f);
    T drdx = x / r;
    // Term 1
    T d1 = y * z * (1.0f + drdx) / (x + r + 1e-30f);
    // Term 2
    T u = y * z / (x * r + 1e-30f);
    T du_dx = -(y * z) * (x * x + r2) / (x * x * r * r * r + 1e-30f);
    T atan_u = atan(u);
    T d2 = -x * atan_u - (x * x / 2.0f) * (1.0f / (1.0f + u * u + 1e-30f)) * du_dx;
    // Term 3
    T d3 = z * log(abs(y + r) + 1e-30f) + x * z * drdx / (y + r + 1e-30f);
    // Term 4
    T v = x * z / (y * r + 1e-30f);
    T dv_dx = (z / (y * r + 1e-30f)) * (1.0f - x * x / (r2 + 1e-30f));
    // T atan_v = atan(v);
    T d4 = -(y * y / 2.0f) * (1.0f / (1.0f + v * v + 1e-30f)) * dv_dx;
    // Term 5
    T d5 = y * log(abs(z + r) + 1e-30f) + x * y * drdx / (z + r + 1e-30f);
    // Term 6
    T w = x * y / (z * r + 1e-30f);
    T dw_dx = (y / (z * r + 1e-30f)) * (1.0f - x * x / (r2 + 1e-30f));
    // T atan_w = atan(w);
    T d6 = -(z * z / 2.0f) * (1.0f / (1.0f + w * w + 1e-30f)) * dw_dx;
    return d1 + d2 + d3 + d4 + d5 + d6;
}

template<typename T>
T partial_dy(T x, T y, T z)
{
    T r2 = x * x + y * y + z * z;
    T r = sqrt(r2 + 1e-30f);
    T drdy = y / r;
    // Term 1
    T d1 = z * log(abs(x + r) + 1e-30f) + y * z * drdy / (x + r + 1e-30f);
    // Term 2
    T u = y * z / (x * r + 1e-30f);
    T du_dy = (z / (x * r + 1e-30f)) * (1.0f - y * y / (r2 + 1e-30f));
    // T atan_u = atan(u);
    T d2 = -(x * x / 2.0f) * (1.0f / (1.0f + u * u + 1e-30f)) * du_dy;
    // Term 3
    T d3 = x * z * (1.0f + drdy) / (y + r + 1e-30f);
    // Term 4
    T v = x * z / (y * r + 1e-30f);
    T dv_dy = -(x * z / r) * (1.0f / (y * y + 1e-30f) + 1.0f / (r2 + 1e-30f));
    T atan_v = atan(v);
    T d4 = -y * atan_v - (y * y / 2.0f) * (1.0f / (1.0f + v * v + 1e-30f)) * dv_dy;
    // Term 5
    T d5 = x * log(abs(z + r) + 1e-30f) + x * y * drdy / (z + r + 1e-30f);
    // Term 6
    T w = x * y / (z * r + 1e-30f);
    T dw_dy = (x / (z * r + 1e-30f)) * (1.0f - y * y / (r2 + 1e-30f));
    // T atan_w = atan(w);
    T d6 = -(z * z / 2.0f) * (1.0f / (1.0f + w * w + 1e-30f)) * dw_dy;
    return d1 + d2 + d3 + d4 + d5 + d6;
}

template<typename T>
T partial_dz(T x, T y, T z)
{
    T r2 = x * x + y * y + z * z;
    T r = sqrt(r2 + 1e-30f);
    T drdz = z / r;
    // Term 1
    T d1 = y * log(abs(x + r) + 1e-30f) + y * z * drdz / (x + r + 1e-30f);
    // Term 2
    T u = y * z / (x * r + 1e-30f);
    T du_dz = (y / (x * r + 1e-30f)) * (1.0f - z * z / (r2 + 1e-30f));
    // T atan_u = atan(u);
    T d2 = -(x * x / 2.0f) * (1.0f / (1.0f + u * u + 1e-30f)) * du_dz;
    // Term 3
    T d3 = x * log(abs(y + r) + 1e-30f) + x * z * drdz / (y + r + 1e-30f);
    // Term 4
    T v = x * z / (y * r + 1e-30f);
    T dv_dz = (x / (y * r + 1e-30f)) * (1.0f - z * z / (r2 + 1e-30f));
    // T atan_v = atan(v);
    T d4 = -(y * y / 2.0f) * (1.0f / (1.0f + v * v + 1e-30f)) * dv_dz;
    // Term 5
    T d5 = x * y * (1.0f + drdz) / (z + r + 1e-30f);
    // Term 6
    T w = x * y / (z * r + 1e-30f);
    T dw_dz = -(x * y / r) * (1.0f / (z * z + 1e-30f) + 1.0f / (r2 + 1e-30f));
    T atan_w = atan(w);
    T d6 = -z * atan_w - (z * z / 2.0f) * (1.0f / (1.0f + w * w + 1e-30f)) * dw_dz;
    return d1 + d2 + d3 + d4 + d5 + d6;
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
                T F = primitive(xx, yy, zz);
                sum_phi += sign * F;
                sum_fx += sign * partial_dx(xx, yy, zz);
                sum_fy += sign * partial_dy(xx, yy, zz);
                sum_fz += sign * partial_dz(xx, yy, zz);
            }
        }
    }

    T phi = -rho * sum_phi;
    Eigen::Vector<T, 3> force(-rho * sum_fx, -rho * sum_fy, -rho * sum_fz);

    return { phi, force };
}
