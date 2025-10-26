#include "cosmology.hpp"
#include "cube.hpp"
#include "multipole_cart.hpp"
#include <boost/iostreams/device/mapped_file.hpp>
#include <filesystem>
#include <hdf5.h>

namespace fs = std::filesystem;
using Vector3f = Eigen::Vector3f;

/**
   \example cosmology.cpp
   FMM N-body example program

   It is a rather complex algorithm, roughly based on
   "A short course on fast multiple methods", https://math.nyu.edu/~greengar/shortcourse_fmm.pdf,
   but with some modifications:
   - Multipoles are represented in Cartesian coordinates instead of the usual spherical harmonics.
   - A binary tree is used instead of an octree.

   The parameters are optimise for big GPUs with many registers.
   If you want to run it on smaller GPUs with <256 registers,
   you might want to reduce MULTIPOLE_ORDER to 2 or so. The precision will be worse,
   but at least it will run with usable performance.
 */

PARAMOPT<bool> PRECISION_TEST("precision_test", false);
PARAMOPT<string> IC("ic", "");

PARAMOPT<Tsize_t> NUM_PARTICLES("num_particles", 1000000); // Number of particles
PARAMOPT<double> TREE_FACTOR("tree_factor", 0.5);
PARAMOPT<double> FORCE_TREE_FACTOR("force_tree_factor", 0.05);
PARAMOPT<Tdouble> MAX_DISTFAC("max_distfac", 1.8);
PARAMOPT<Tuint> MAX_NODESIZE("max_nodesize", 800);
PARAMOPT<Tdouble> DT("dt", 5E-3);

template<class T>
Vector<T, 4> color(T pot)
{
    T pc = log2(clamp((-pot - 0.0f) * 0.6f, 1, 15.99f));

    gpu_float slot = floor(pc);
    gpu_float x = pc - slot;
    gpu_assert(slot >= 0);
    gpu_assert(slot < 4);
    Vector<T, 4> ret =
        cond(slot == 0,
             Vector<gpu_float, 4>({ 0, x, 1 - x, 0 }),
             cond(slot == 1,
                  Vector<gpu_float, 4>({ x, 1 - x, 0, 0 }),
                  cond(slot == 2, Vector<gpu_float, 4>({ 1, x, 0, 0 }), Vector<gpu_float, 4>({ 1, 1, x, 0 }))));
    return ret;
}

template<typename M>
void test()
{
    using T = typename M::value_type;

    cout.precision(10);
    cout << "\ntest. M=" << pretty_typename(typeid(M)) << endl;
    T mass = 13;

    vector<Vector<T, 3>> rtab = { { 0.32, 0.13, -0.78 }, { 0.8, -0.7, -0.2 } };
    auto m = M::zero();

    for (auto r : rtab)
    {
        m += M::from_particle(r, 1);
    }

    cout << "m=" << m << endl;

    Vector<T, 3> shift_ext = { -0.76, 1.1, 0.99 };
    m = m.shift_ext(shift_ext);

    cout << "shift_ext: " << m << endl;

    ofstream PLOT("plot-" + to_string(M::order));
    for (double ds = 0.1; ds < 1E12; ds *= 1.1)
    {
        Vector<T, 3> shift = Vector<T, 3>{ 4.3, 3.7, -5.2 } * ds;
        Vector<T, 3> r = { 0.5, 0.6, 0.3 };
        Vector<T, 3> shift2 = { 0.3, 0.22, -0.53 };

        T Pist = 0;
        Vector<T, 3> Fist = { 0, 0, 0 };
        for (auto& p : rtab)
        {
            Vector<T, 3> dist = p - shift_ext - (r + shift);
            Pist += -1.0 / dist.norm();
            Fist += dist / dist.norm() / dist.squaredNorm();
        }

        auto m2 = m.makelocal(shift);
        // cout << "makelocal: " << m2 << endl;

        auto m3 = m2.shift_loc(shift2);
        auto m4 = m3.rot();

        T P = m4.calc_loc_potential((rot(r) - rot(shift2)).eval());
        // cout << "P=" << P << endl;

        Vector<T, 3> F = rot(m4.calc_force((rot(r) - rot(shift2)).eval()), -1);
        // cout << "F=" << F << endl;
        cout << "ds=" << ds << ", P=" << P << ", Pist=" << Pist << ", err=" << (P - Pist) / abs(Pist) << ", F=" << F
             << ", Fist=" << Fist << ", err=" << (F - Fist).norm() / Fist.norm() << endl;
        PLOT << ds << " " << (F - Fist).norm() / Fist.norm() << " " << std::abs((P - Pist) / abs(Pist)) << endl;
    }

    // M loc = m.makelocal({2.3, 0.8, -0.4});
    // cout << "loc=" << loc << endl;
}

constexpr double m_ = 1.0 / (15 * 3.085678e22) / 1.5;  // 100Mpc==1
constexpr double s_ = 1.0 / (365.2422 * 86400) / 1E11; // Gyear
// constexpr double kg_ = 1.0 / 1.98892e30 / 1E17;        // 1E10 solar mass
constexpr double kg_ = 6.67259E-11 * m_ * m_ * m_ / s_ / s_; // G_=1

// constexpr double kg_ = 1;
// constexpr double m_ = 1;
// constexpr double s_ = 1;
constexpr double pc_ = 3.085678e16 * m_;
constexpr double a_ = 1;
constexpr double yr_ = 365.2422 * 86400 * s_;

constexpr double kpc_ = 1000 * pc_;
constexpr double Mpc_ = 1000 * kpc_;
constexpr double km_ = 1000 * m_;

constexpr double g_ = kg_ / 1000;
constexpr double cm_ = m_ / 100;
constexpr double erg_ = g_ * cm_ * cm_ / s_ / s_;
constexpr double G_ = 6.67259E-8 * erg_ / g_ / g_ * cm_; // gravitational constant G
static_assert(abs(G_ - 1) < 1E-10);

bool is_cosmic = false;
struct
{
    Tdouble H0;
    Tdouble omega_m;
    Tdouble omega_lambda;
    Tdouble box_size = 0;
    Tdouble density;
    Tdouble a = 1;
    Tdouble t = 0;
    Tdouble today;

    void shift(double dt)
    {
        if (is_cosmic)
        {
            t += dt;
            a += H0 * a * sqrt(omega_m * pow<-3, 1>(a) + omega_lambda) * dt;
        }
        else
        {
            t += dt;
        }
    }

} cosmic;

void read_music2_hdf5(const std::string& filename,
                      std::vector<Vector<Tfloat, 3>>& positions,
                      std::vector<Vector<Tfloat, 3>>& velocities,
                      Tfloat& mass,
                      int part_type = 1)
{
    // constexpr float boxlength = 100*1000;
    is_cosmic = true;

    positions.clear();
    velocities.clear();

    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0)
        return; // Error opening file

    // Open Header group and read NumPart_ThisFile
    hid_t header_id = H5Gopen(file_id, "/Header", H5P_DEFAULT);
    if (header_id < 0)
    {
        H5Fclose(file_id);
        return;
    }

    auto read_double_attr = [&](const char* name) -> double {
        double value = 0.0;
        hid_t attr_id = H5Aopen(header_id, name, H5P_DEFAULT);
        if (attr_id >= 0)
        {
            H5Aread(attr_id, H5T_NATIVE_DOUBLE, &value);
            H5Aclose(attr_id);
        }
        else
        {
            std::cerr << "Warning: Attribute " << name << " not found" << std::endl;
        }
        return value;
    };

    double hubble_param = read_double_attr("HubbleParam");
    cosmic.H0 = hubble_param * 100.0 * km_ / s_ / Mpc_; // in km/s/Mpc
    cosmic.omega_m = read_double_attr("Omega0");
    cosmic.omega_lambda = read_double_attr("OmegaLambda");
    double redshift = read_double_attr("Redshift");
    cosmic.box_size = read_double_attr("BoxSize") * kpc_ / hubble_param / a_;
    cosmic.a = 1.0 / (1 + redshift);

    {
        double da = 1E-6;
        cosmic.t = 0;
        for (double a = 0.5 * da; a < cosmic.a; a += da)
        {
            cosmic.t += 1 / cosmic.H0 * da / (a * sqrt(cosmic.omega_m * pow<-3, 1>(a) + cosmic.omega_lambda));
        }
    }

    // double sigma8 = read_double_attr("Sigma8");  // Optional, may not be present

    // H5Gclose(header_id);
    // H5Fclose(file_id);

    {
        auto c = cosmic;
        while (c.a < 1)
        {
            c.shift(1E-5);
        }
        cosmic.today = c.t;
    }

    std::cout << "H0: " << cosmic.H0 << " = " << cosmic.H0 / (km_ / s_ / Mpc_) << " km/s/Mpc" << std::endl;
    std::cout << "Omega_m: " << cosmic.omega_m << std::endl;
    std::cout << "OmegaLambda: " << cosmic.omega_lambda << std::endl;
    std::cout << "Redshift: " << redshift << std::endl;
    std::cout << "BoxSize: " << cosmic.box_size << " = " << cosmic.box_size / Mpc_ << " Mpc" << std::endl;
    // if (sigma8 != 0.0) std::cout << "Sigma8: " << sigma8 << std::endl;
    cout << "G=" << G_ << endl;

    hid_t attr_id = H5Aopen(header_id, "NumPart_ThisFile", H5P_DEFAULT);
    if (attr_id < 0)
    {
        H5Gclose(header_id);
        H5Fclose(file_id);
        return;
    }
    hid_t space_id = H5Aget_space(attr_id);
    uint32_t npart[6];
    H5Aread(attr_id, H5T_NATIVE_UINT32, npart);
    H5Sclose(space_id);
    H5Aclose(attr_id);
    H5Gclose(header_id);

    size_t N = npart[part_type];
    if (N == 0)
    {
        H5Fclose(file_id);
        return;
    }

    std::string group_name = "/PartType" + std::to_string(part_type);
    hid_t group_id = H5Gopen(file_id, group_name.c_str(), H5P_DEFAULT);
    if (group_id < 0)
    {
        H5Fclose(file_id);
        return;
    }

    // Read Coordinates
    hid_t dset_pos = H5Dopen(group_id, "Coordinates", H5P_DEFAULT);
    if (dset_pos < 0)
    {
        H5Gclose(group_id);
        H5Fclose(file_id);
        return;
    }
    hid_t dtype_pos = H5Dget_type(dset_pos);
    bool is_double = (H5Tget_class(dtype_pos) == H5T_FLOAT && H5Tget_size(dtype_pos) == 8);
    hid_t native_type = is_double ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;

    std::vector<float> pos_data(3 * N);
    if (is_double)
    {
        std::vector<double> temp(3 * N);
        H5Dread(dset_pos, native_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp.data());
        for (size_t i = 0; i < 3 * N; ++i)
            pos_data[i] = static_cast<float>(temp[i]);
    }
    else
    {
        H5Dread(dset_pos, native_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, pos_data.data());
    }
    H5Tclose(dtype_pos);
    H5Dclose(dset_pos);

    // Read Velocities (similar to positions)
    hid_t dset_vel = H5Dopen(group_id, "Velocities", H5P_DEFAULT);
    if (dset_vel < 0)
    {
        H5Gclose(group_id);
        H5Fclose(file_id);
        return;
    }
    hid_t dtype_vel = H5Dget_type(dset_vel);
    // Assume same type as positions for simplicity
    std::vector<float> vel_data(3 * N);
    if (is_double)
    {
        std::vector<double> temp(3 * N);
        H5Dread(dset_vel, native_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp.data());
        for (size_t i = 0; i < 3 * N; ++i)
            vel_data[i] = static_cast<float>(temp[i]);
    }
    else
    {
        H5Dread(dset_vel, native_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, vel_data.data());
    }
    H5Tclose(dtype_vel);
    H5Dclose(dset_vel);

    H5Gclose(group_id);
    H5Fclose(file_id);

    // Populate output vectors
    positions.reserve(N);
    velocities.reserve(N);
    for (size_t i = 0; i < N; ++i)
    {
        size_t idx = 3 * i;
        positions.emplace_back(pos_data[idx], pos_data[idx + 1], pos_data[idx + 2]);
        velocities.emplace_back(vel_data[idx], vel_data[idx + 1], vel_data[idx + 2]);
    }

    /**
       positions have units m_/a_
       velocities have units m_/s_*a_
     */

    for (auto& x : positions)
    {
        const double h = cosmic.H0 / (100 * km_ / s_ / Mpc_);
        x *= kpc_ / a_ / h;
        for (auto& xx : x)
        {
            xx -= 0.5f * cosmic.box_size;
        }
    }
    for (auto& v : velocities)
    {
        v *= sqrt(cosmic.a) * km_ / s_;
        v *= cosmic.a;
    }
    cosmic.density = cosmic.omega_m * 3 * pow2(cosmic.H0) / (8 * PI * G_) * pow3(a_);
    mass = cosmic.density * pow3(cosmic.box_size) / N;
}

template<typename T>
void generate_IC_HDF5(CosmosData<T>& cosmos, filesystem::path fn)
{
    vector<Vector<Tfloat, 3>> pos, vel;
    Tfloat mass;
    read_music2_hdf5(fn, pos, vel, mass);

    cout << "pos.size()=" << pos.size() << endl;
    cout << "vel.size()=" << vel.size() << endl;

    if (pos.size() != cosmos.x.size())
    {
        cout << "Number of particles does not match. pos.size()=" << pos.size()
             << ", cosmos.num_particles=" << cosmos.x.size() << endl;
        exit(0);
    }

    cosmos.x = std::move(pos);
    cosmos.v = std::move(vel);
#if CONSTANT_MASS
    cosmos.constantMass = mass;
#else
    cosmos.mass.fill(mass);
#endif
}

template<typename T>
void generate_IC(CosmosData<T>& cosmos, const char* filename = nullptr)
{
    cout << "generating initial conditions..." << flush;
    size_t N = cosmos.x.size();

    std::default_random_engine generator;
    std::normal_distribution<double> distribution;
    std::uniform_real_distribution<double> distribution2;

    if (filename)
    {
        cosmos.v.fill({ 0, 0, 0 });
        cout << "Reading from file " << filename << endl;
#if !WITH_OPENCV
        throw std::runtime_error("Need opencv to read images");
#else
        cv::Mat image_color = cv::imread(filename);
        if (image_color.empty())
        {
            throw std::runtime_error("Failed to read image");
        }

        cv::Mat image_gray;
        cv::cvtColor(image_color, image_gray, cv::COLOR_BGR2GRAY);

        uint max_extent = max(image_gray.rows, image_gray.cols);
        Vector<double, 3> cm = { 0, 0, 0 };
        buffer_map cx(cosmos.x);
        for (auto& r : cx)
        {
            // cout << "." << flush;
            while (true)
            {
                for (auto& xx : r)
                {
                    xx = distribution2(generator);
                }
                r[2] *= 0.1f;
                Vector<int, 3> ri = (r * max_extent).template cast<int>();
                if (ri[0] < image_gray.cols && ri[1] < image_gray.rows)
                {
                    uint8_t c = image_gray.at<uint8_t>(
                        { static_cast<int>(r[0] * max_extent), static_cast<int>(r[1] * max_extent) });
                    if (distribution2(generator) * 255 < c)
                    {
                        cm += r.template cast<double>();
                        break;
                    }
                }
            }
        }
        cm /= N;
        for (auto& r : cx)
        {
            r -= cm.cast<Tfloat>();
        }
        double extent2 = 0;
        for (auto& r : cx)
        {
            extent2 += r.squaredNorm();
        }
        extent2 /= N;
        for (auto& r : cx)
        {
            r *= 0.5 / sqrt(extent2);
            r[1] *= -1;
        }
#endif
    }
    else
    {
        Tint MODE = 2;
        if (MODE == 2)
        {
            buffer_map x(cosmos.x);
            buffer_map v(cosmos.v);
            for (Tuint k = 0; k < N; ++k) // Setting the initial conditions:
            {                             // N particles of mass 1/N each are randomly placed in a sphere of radius 1
                Vector<T, 3> xk;
                Vector<T, 3> vk;
                do
                {
                    for (Tuint i = 0; i < 3; ++i)
                    {
                        xk[i] = distribution(generator) * 0.2;
                        vk[i] = distribution(generator) * 0.2;
                    }
                } while (xk.squaredNorm() >= 1);
                x[k] = xk;
                vk += Vector<T, 3>({ -xk[1], xk[0], 0 }) / (Vector<T, 3>({ -xk[1], xk[0], 0 })).norm() * 0.4f
                      * min(xk.norm() * 10, (T)1);
                if (k < N / 2)
                    vk = -vk;
                v[k] = vk;
                if (k < N / 2)
                {
                    x[k] += Vector<T, 3>{ 0.8, 0.2, 0.0 };
                    v[k] += Vector<T, 3>{ -0.4, 0.0, 0.0 };
                }
                else
                {
                    x[k] -= Vector<T, 3>{ 0.8, 0.2, 0.0 };
                    v[k] += Vector<T, 3>{ 0.4, 0.0, 0.0 };
                }
            }
        }
        else if (MODE == 3)
        {
            buffer_map x(cosmos.x);
            for (Tsize_t p = 0; p < cosmos.x.size(); ++p)
            {
                for (Tint k = 0; k < 3; ++k)
                {
                    do
                    {
                        x[p][k] = distribution(generator);
                    } while (abs(x[p][k]) >= 1);
                }
            }
            cosmos.v.fill({ 0, 0, 0 });
        }
    }
#if CONSTANT_MASS
    cosmos.constantMass = 1.0 / N;
#else
    cosmos.mass.fill(1.0 / N);
#endif
    cout << "ok" << endl;
}

int main(int argc, char** argv)
{
    cout << "sizeof(matter_multipole)=" << sizeof(Cosmos<Tfloat, MULTIPOLE_ORDER>::matter_multipole) << endl;
    cout << "sizeof(force_multipole)=" << sizeof(Cosmos<Tfloat, MULTIPOLE_ORDER>::force_multipole) << endl;
    cout << "alignof(matter_multipole)=" << alignof(Cosmos<Tfloat, MULTIPOLE_ORDER>::matter_multipole) << endl;
    cout << "alignof(force_multipole)=" << alignof(Cosmos<Tfloat, MULTIPOLE_ORDER>::force_multipole) << endl;

    init_params(argc, argv);

    if constexpr (false)
    {
        using T = debugtype<double>;

        test<multipole<T>>();
        test<multipole<T, T>>();
        test<multipole<T, T, T>>();
        test<multipole<T, T, T, T>>();
        test<multipole<T, T, T, T, T>>();

        return 0;
    }
    else
    {
        unique_ptr<sdl_window> window = sdl_window::create("fmm nbody",
                                                           { 1024, 768 },
                                                           SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY
#if GOOPAX_VERSION_ID < 50802
                                                           ,
                                                           static_cast<goopax::envmode>(env_ALL & ~env_VULKAN)
#endif
        );

        goopax_device device = default_device(env_CUDA);
        if (!device.valid())
        {
            device = window->device;
        }

#if GOOPAX_DEBUG
        // Increasing number of threads to have meaningful race condition checks.
        device.force_global_size(192);
#endif

#if WITH_OPENGL
        optional<opengl_buffer<Vector3<Tfloat>>> x_gl;
        optional<opengl_buffer<Vector4<Tfloat>>> color_gl;
        kernel<void(const buffer<Vector<Tfloat, 3>>& cx, const buffer<Tfloat>& potential)> set_colors;
#endif

#if WITH_METAL
        unique_ptr<particle_renderer> metalRenderer;
#endif
#if WITH_VULKAN
        CosmosData<Tfloat> vulkanData;
        unique_ptr<goopax_draw::vulkan::Renderer> vulkanRenderer;
#endif

        CosmosData<Tfloat> initData;

        if (false)
        {
        }
#if WITH_METAL
        else if (auto* m = dynamic_cast<sdl_window_metal*>(&*window))
        {
            metalRenderer = make_unique<particle_renderer>(dynamic_cast<sdl_window_metal&>(*window));
            //            x.assign(device, NUM_PARTICLES());
            //          color.assign(device, NUM_PARTICLES());
        }
#endif
#if WITH_VULKAN && GOOPAX_VERSION_ID >= 50802
        else if (auto* v = dynamic_cast<sdl_window_vulkan*>(&*window))
        {
            backend_create_params params = {
                .vulkan = { .usage_bits = VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT
                                          | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
                                          | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR }
            };
            if (device.get_envmode() == env_CUDA)
            {
                params.vulkan.vkExternalMemoryHandleTypeFlags = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
            }
            vulkanData.x.assign(window->device, NUM_PARTICLES(), params);
            vulkanData.v.assign(window->device, NUM_PARTICLES(), params);
            vulkanData.force.assign(window->device, NUM_PARTICLES(), params);
            vulkanData.potential.assign(window->device, NUM_PARTICLES(), params);
#if !CONSTANT_MASS
            vulkanData.mass.assign(window->device, NUM_PARTICLES(), params);
#endif
            vulkanData.tmps.assign(window->device, NUM_PARTICLES(), params);

            if (device.get_envmode() == env_CUDA)
            {
                auto link = [&](auto& vulkan, auto& cuda) {
                    int fd;
                    VkMemoryGetFdInfoKHR getFdInfo = { .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
                                                       .memory = get_vulkan_device_memory(vulkan),
                                                       .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT };
                    call_vulkan(v->vkGetMemoryFdKHR(v->vkDevice, &getFdInfo, &fd));

                    cuda = vulkan.create_from_external_handle(device, fd, NUM_PARTICLES());
                };

                link(vulkanData.x, initData.x);
                link(vulkanData.v, initData.v);
                link(vulkanData.force, initData.force);
                link(vulkanData.potential, initData.potential);
#if !CONSTANT_MASS
                link(vulkanData.mass, initData.mass);
#endif
                link(vulkanData.tmps, initData.tmps);
            }
            else
            {
                swap(vulkanData, initData);
            }
        }
#endif
#if WITH_OPENGL
        else if (dynamic_cast<sdl_window_gl*>(&*window))
        {
            x_gl = opengl_buffer<Vector3<Tfloat>>(device, NUM_PARTICLES());
            color_gl = opengl_buffer<Vector4<Tfloat>>(device, NUM_PARTICLES());

            set_colors.assign(device, [&](const resource<Vector<Tfloat, 3>>& cx, const resource<Tfloat>& potential) {
                gpu_for_global(0, x_gl->size(), [&](gpu_uint k) {
                    (*color_gl)[k] = ::color(potential[k]);
                    (*x_gl)[k] = cx[k];
                    // Tweaking z coordinate to use potential for depth testing.
                    // Particles are displayed according to their x and y coordinates.
                    // If multiple particles are drawn at the same pixel, the one with the
                    // highest potential will be shown.
                    (*x_gl)[k][2] = -potential[k] * 0.01f;
                });
            });
        }
#endif
        else
        {
            initData.x.assign(device, NUM_PARTICLES);
            initData.v.assign(device, NUM_PARTICLES);
            initData.force.assign(device, NUM_PARTICLES);
            initData.potential.assign(device, NUM_PARTICLES);
#if !CONSTANT_MASS
            initData.mass.assign(device, NUM_PARTICLES);
#endif
            initData.tmps.assign(device, NUM_PARTICLES);
        }

        using T = Tfloat;

        size_t tree_size = NUM_PARTICLES() * TREE_FACTOR() + 100000 + (2 << min_tree_depth);
        size_t min_force_tree_size = tree_size * FORCE_TREE_FACTOR();

        if (argc >= 2)
        {
            fs::path fn = argv[1];
            cout << "fn=" << fn << ", ext=" << fn.extension();
            if (fn.extension() == ".hdf5")
            {
                generate_IC_HDF5(initData, fn);
            }
            else
            {
                generate_IC(initData, argv[1]);
            }
        }
        else
        {
            generate_IC(initData);
        }

        Cosmos<T, MULTIPOLE_ORDER> cosmos(std::move(initData),

                                          tree_size,
                                          min_force_tree_size,
                                          MAX_DISTFAC(),
                                          MAX_NODESIZE());

#if WITH_VULKAN && GOOPAX_VERSION_ID >= 50802
        if (auto* v = dynamic_cast<sdl_window_vulkan*>(&*window))
        {
            vulkanRenderer = make_unique<goopax_draw::vulkan::Renderer>(
                *v, cosmic.box_size / 2, std::array<unsigned int, 2>{ 1000, 500 });
        }
#endif

        array<kernel<void(
                  const buffer<Vector<Tfloat, 3>>& x, buffer<Vector<Tfloat, 3>>& force, buffer<Tfloat>& potential)>,
              2>
            add_cube_force;

        kernel<void(buffer<Tfloat> & potential)> adjust_palette;

        if (is_cosmic)
        {
            for (bool with_pot : { false, true })
            {
                add_cube_force[with_pot].assign(
                    device,
                    [with_pot](const resource<Vector<Tfloat, 3>>& x,
                               resource<Vector<Tfloat, 3>>& force,
                               resource<Tfloat>& potential) {
                        gpu_for_global(0, x.size(), [&](gpu_uint k) {
                            auto [P, F] = get_potential_and_force<gpu_float>(x[k], cosmic.box_size, -cosmic.density);

                            gpu_assert(isfinite(P));
                            gpu_assert(isfinite(F.sum()));

                            force[k] += F;
                            if (with_pot)
                            {
                                potential[k] += P;

                                potential[k] -= static_cast<float>(1E20 * G_ * kg_ / m_);
                                potential[k] = log10(clamp(-potential[k] * 60.f, 1.01f, 99.9f)) * (1.f / 2);

                                gpu_assert(potential[k] > 0);
                                gpu_assert(potential[k] <= 1);
                            }
                        });
                    });
            }
        }
        else
        {
            adjust_palette.assign(device, [](resource<Tfloat>& potential) {
                for_each_global(potential, [](auto& p) {
                    p = log10(clamp(-p * 2.f, 1.01f, 99.9f)) * (1.f / 2);
                    gpu_assert(p > 0);
                    gpu_assert(p <= 1);
                });
            });
        }

        auto last_fps_time = steady_clock::now();
        size_t last_fps_step = 0;

        bool quit = false;
        float distance = 2;
        float theta = 0;
        Vector<float, 2> last_mouse;
        int mouse_button_down = 0;
        Vector<float, 2> xypos = { 0, 0 };

        constexpr uint make_tree_every = 2;
        constexpr uint render_every = 1;
        constexpr uint pot_every = 4;

        cosmos.make_initial_tree();
#if WITH_VULKAN
        vulkanData.swapBuffers(false);
#endif

        float rate = 0;

        for (size_t step = 0; !quit; ++step)
        {
            cout << "step=" << step << ". a=" << cosmic.a << ", t=" << cosmic.t / (1E9 * yr_) << " Gyr" << endl;
            while (auto e = window->get_event())
            {
                if (e->type == SDL_EVENT_QUIT)
                {
                    quit = true;
                }
                else if (e->type == SDL_EVENT_KEY_DOWN)
                {
                    switch (e->key.key)
                    {
                        case SDLK_ESCAPE:
                            quit = true;
                            break;
                        case SDLK_F:
                            window->toggle_fullscreen();
                            break;
                    };
                }
                else if (e->type == SDL_EVENT_MOUSE_BUTTON_DOWN)
                {
                    SDL_GetMouseState(&last_mouse[0], &last_mouse[1]);
                    mouse_button_down = e->button.button;
                    cout << "button: " << (int)e->button.button << endl;
                }
                else if (e->type == SDL_EVENT_MOUSE_BUTTON_UP)
                {
                    mouse_button_down = 0;
                }
                else if (e->type == SDL_EVENT_MOUSE_WHEEL)
                {
                    distance *= exp(-0.1f * e->wheel.y);
                }
            }
            if (mouse_button_down)
            {
                Vector<float, 2> mouse;
                SDL_GetMouseState(&mouse[0], &mouse[1]);
                if (mouse_button_down == 1)
                {
                    theta += (mouse[0] - last_mouse[0]) * 0.01f;
                }
                else if (mouse_button_down == 3)
                {
                    xypos -= Vector<float, 2>{ mouse[0] - last_mouse[0], mouse[1] - last_mouse[1] } * 0.004f;
                }
                last_mouse = mouse;
            }

            if (step != 0)
            {
#if WITH_VULKAN
                if (vulkanRenderer.get() != nullptr)
                {
                    auto& window = vulkanRenderer->window;
                    window.vkWaitForFences(window.vkDevice, 1, &vulkanRenderer->inFlightFence, VK_TRUE, window.timeout);
                }
#endif
            }

            cosmic.shift(0.25 * DT());
            cosmos.movefunc(0.5 * DT() / pow2(cosmic.a), cosmos.v, cosmos.x);
            cosmic.shift(0.25 * DT());

            cosmos.update_tree(step % pot_every != 0);
#if WITH_VULKAN
            vulkanData.swapBuffers(step % pot_every != 0);
#endif

            if (step % make_tree_every == 0)
            {
                cout << "reconstructing tree" << endl;
                cosmos.make_tree();
            }

            // cout << "x=" << Cosmos->x << endl;
            // cout << "v=" << Cosmos->v << endl;
            // cout << "mass=" << Cosmos->mass << endl;
            cosmos.compute_force(step % pot_every == 0);

            if (PRECISION_TEST() && step / make_tree_every % 128 == 0)
            {
                cosmos.precision_test();
                // exit(0);
            }
#if WITH_TIMINGS
            device.wait_all();
            auto t0 = steady_clock::now();
#endif

            if (is_cosmic)
            {
                add_cube_force[step % pot_every == 0](cosmos.x, cosmos.force, cosmos.potential);
            }
            else if (step % pot_every == 0)
            {
                adjust_palette(cosmos.potential);
            }
#if WITH_TIMINGS
            device.wait_all();
            auto t1 = steady_clock::now();
#endif
            goopax_future<void> compute_future = device.enqueue_fence();

            cosmos.kick(DT() * pow<-1, 1>(cosmic.a) * G_, cosmos.force, cosmos.v);
#if WITH_TIMINGS
            device.wait_all();
            auto t2 = steady_clock::now();
#endif

            compute_future.wait();
            auto now = steady_clock::now();
            // if (now - last_fps_time > chrono::seconds(1) || true)
            {
                if (step % max(pot_every, make_tree_every) == 0)
                {
                    rate = (step - last_fps_step) / chrono::duration<float>(now - last_fps_time).count();
                    last_fps_step = step;
                    last_fps_time = now;
                }

                stringstream ss;

                if (is_cosmic)
                {
                    double caltime = cosmic.t - cosmic.today + 2025 * yr_;

                    ss << "N-Body Simulation (Fast Multipole)" << endl
                       << "device: " << device.name() << endl
                       << "number of particles: " << cosmos.x.size() << endl
                       << "step: " << step << endl
                       << "steps per second: " << rate << endl
                       << "time: " << cosmic.t / (1E9 * yr_) << " Gyr"
                       << " (" << static_cast<ssize_t>(abs(caltime + (caltime >= 0 ? 1.0 : 0.0)) / yr_)
                       << ((cosmic.t - cosmic.today + 2025 * yr_ > 0) ? " AD)" : " BC)") << endl
                       << "scale factor: " << cosmic.a << " (z=" << 1 / cosmic.a - 1 << ")" << endl
                       << endl;
                }
                else
                {
                    ss << "N-Body Simulation" << endl
                       << "Fast Multipole (" << MULTIPOLE_ORDER << "th order)" << endl
                       << "device: " << device.name() << endl
                       << "number of particles: " << cosmos.x.size() << endl
                       << "simulation step: " << step << endl
                       << "fps: " << rate << endl
                       << endl;
                }

#if WITH_VULKAN
                if (vulkanRenderer.get())
                {
                    auto size = window->get_size();
                    vulkanRenderer->updateText(ss.str(), { size[0] - 1000, size[1] - 600 }, { 600, 500 }, 55);
                }
#endif
            }

            if (step % render_every == 0)
            {
                if (false)
                {
                }
#if WITH_METAL
                else if (metalRenderer.get())
                {
                    metalRenderer->render(Cosmos.x);
                }
#endif

#if WITH_VULKAN && GOOPAX_VERSION_ID >= 50802
                else if (vulkanRenderer.get())
                {
                    if (device.get_envmode() == env_CUDA)
                    {
                        vulkanRenderer->render(vulkanData.x, vulkanData.potential, distance, theta, xypos);
                    }
                    else
                    {
                        vulkanRenderer->render(cosmos.x, cosmos.potential, distance, theta, xypos);
                    }
                }
#endif
#if WITH_OPENGL
                else if (x_gl)
                {
                    set_colors(cosmos.x, cosmos.potential);
                    render(window->window, *x_gl, &*color_gl);
                    SDL_GL_SwapWindow(window->window);
                }
#endif
                else
                {
                    cout << "x=" << const_buffer_map(cosmos.x, 0, 10) << "..." << endl;
                }
            }
#if WITH_TIMINGS
            device.wait_all();
            auto t3 = steady_clock::now();
#endif
            cosmic.shift(0.25 * DT());
            cosmos.movefunc(0.5 * DT() / cosmic.a, cosmos.v, cosmos.x);
            cosmic.shift(0.25 * DT());
#if WITH_TIMINGS
            device.wait_all();
            auto t4 = steady_clock::now();
#endif

#if WITH_TIMINGS
            // cout << "treecount: " << duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << endl;
            cout << "main loop:\n"
                 << "  add_cube_force: " << duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << endl
                 << "  kick: " << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << endl
                 << "  render: " << duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " ms" << endl
                 << "  move: " << duration_cast<std::chrono::milliseconds>(t4 - t3).count() << " ms" << endl
                 << endl;
#endif
        }
        return 0;
    }
}
