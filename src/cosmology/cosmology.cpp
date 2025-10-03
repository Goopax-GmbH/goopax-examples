#include "cosmology.hpp"
#include "multipole_cart.hpp"

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
PARAMOPT<Tdouble> MAX_DISTFAC("max_distfac", 1.8);
PARAMOPT<Tuint> MAX_NODESIZE("max_nodesize", 800);
PARAMOPT<Tdouble> DT("dt", 5E-3);

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

int main(int argc, char** argv)
{
    cout << "sizeof(matter_multipole)=" << sizeof(cosmos<Tfloat, MULTIPOLE_ORDER>::matter_multipole) << endl;
    cout << "sizeof(force_multipole)=" << sizeof(cosmos<Tfloat, MULTIPOLE_ORDER>::force_multipole) << endl;
    cout << "alignof(matter_multipole)=" << alignof(cosmos<Tfloat, MULTIPOLE_ORDER>::matter_multipole) << endl;
    cout << "alignof(force_multipole)=" << alignof(cosmos<Tfloat, MULTIPOLE_ORDER>::force_multipole) << endl;

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
        goopax_device device = window->device;

#if GOOPAX_DEBUG
        // Increasing number of threads to have meaningful race condition checks.
        device.force_global_size(192);
#endif

#if WITH_METAL
        particle_renderer Renderer(dynamic_cast<sdl_window_metal&>(*window));
        buffer<Vector3<Tfloat>> x(device, NUM_PARTICLES()); // OpenGL buffer
        buffer<Vector4<Tfloat>> color(device, NUM_PARTICLES());
#elif WITH_VULKAN && GOOPAX_VERSION_ID >= 50802

        backend_create_params params = { .vulkan = { .usage_bits = VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT
                                                                   | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT
                                                                   | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
                                                                   | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR } };

        // buffer<Vector<Tfloat, 3>> x(window->device, NUM_PARTICLES(), params);
        // buffer<Tfloat> color(window->device, NUM_PARTICLES(), params);
        //  ranges::fill(buffer_map(color), Vector<float, 4>{ 1.f, 0.6f, 0.7f, 1.0f });

        VulkanRenderer Renderer(dynamic_cast<sdl_window_vulkan&>(*window));
#elif WITH_OPENGL
        opengl_buffer<Vector3<Tfloat>> x(device, NUM_PARTICLES()); // OpenGL buffer
        opengl_buffer<Vector4<Tfloat>> color(device, NUM_PARTICLES());
#else
        buffer<Vector3<Tfloat>> x(device, NUM_PARTICLES());
        buffer<Vector4<Tfloat>> color(device, NUM_PARTICLES());
#endif

        using T = Tfloat;

        size_t tree_size = NUM_PARTICLES() * TREE_FACTOR() + 100000;

        cosmos<T, MULTIPOLE_ORDER> Cosmos(device, NUM_PARTICLES(), tree_size, MAX_DISTFAC(), MAX_NODESIZE(), params);

        if (argc >= 2)
        {
            Cosmos.make_IC(argv[1]);
        }

        /*        kernel set_colors(device, [&](const resource<Vector<T, 3>>& cx) {
                gpu_for_global(0, x.size(), [&](gpu_uint k) {
              color[k] = Cosmos.potential2[k];//::color(static_cast<gpu_float>(Cosmos.potential2[k]));
                    x[k] = cx[k].cast<gpu_float>();
                    // Tweaking z coordinate to use potential for depth testing.
                    // Particles are displayed according to their x and y coordinates.
                    // If multiple particles are drawn at the same pixel, the one with the
                    // highest potential will be shown.
                    //x[k][2] = -0.9f + static_cast<gpu_float>(-Cosmos.potential2[k]) * 0.001f;
                });
            });
        */

        auto last_fps_time = steady_clock::now();
        size_t last_fps_step = 0;

        bool quit = false;
        constexpr uint make_tree_every = 4;
        constexpr uint render_every = 4;

        Cosmos.make_tree();

        Cosmos.movefunc(0.5f * DT(), Cosmos.v, Cosmos.x);

        for (size_t step = 0; !quit; ++step)
        {
            cout << "step=" << step << endl;
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
            }

            Cosmos.update_tree();

            if (step % make_tree_every == 0)
            {
                cout << "reconstructing tree" << endl;
                Cosmos.make_tree();
            }

            // cout << "x=" << Cosmos->x << endl;
            // cout << "v=" << Cosmos->v << endl;
            // cout << "mass=" << Cosmos->mass << endl;
            Cosmos.compute_force(step % render_every == 0);

            if (PRECISION_TEST() && step / make_tree_every % 128 == 0)
            {
                Cosmos.precision_test();
                // exit(0);
            }

            Cosmos.kick(DT(), Cosmos.force, Cosmos.v);

            auto now = steady_clock::now();
            if (now - last_fps_time > chrono::seconds(1))
            {
                stringstream title;
                Tdouble rate = (step - last_fps_step) / chrono::duration<double>(now - last_fps_time).count();
                title << "N-body. N=" << Cosmos.x.size() << ", step " << step << ", " << rate
                      << " fps, device=" << device.name();
                window->set_title(title.str());
                last_fps_step = step;
                last_fps_time = now;
            }

            if (step % render_every == 0)
            {
                Cosmos.movefunc(0.5 * DT(), Cosmos.v, Cosmos.x);
                // set_colors(Cosmos.x);

#if WITH_METAL
                Renderer.render(x);
#elif WITH_VULKAN && GOOPAX_VERSION_ID >= 50802
                Renderer.render(Cosmos.x, Cosmos.potential);
#elif WITH_OPENGL
                render(window->window, x, &color);
                SDL_GL_SwapWindow(window->window);
#else
                cout << "x=" << x << endl;
#endif
                Cosmos.movefunc(0.5 * DT(), Cosmos.v, Cosmos.x);
            }
            else
            {
                Cosmos.movefunc(DT(), Cosmos.v, Cosmos.x);
            }
        }
        return 0;
    }
}
