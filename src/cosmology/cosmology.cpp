#include "cosmology.hpp"

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

int main(int argc, char** argv)
{
    init_params(argc, argv);

    unique_ptr<sdl_window> window = sdl_window::create("fmm nbody",
                                                       { 1024, 768 },
                                                       SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY,
                                                       static_cast<goopax::envmode>(env_ALL & ~env_VULKAN));
    goopax_device device = window->device;

#if GOOPAX_DEBUG
    // Increasing number of threads to be able to check for race conditions.
    device.force_global_size(192);
#endif

#if WITH_METAL
    particle_renderer Renderer(dynamic_cast<sdl_window_metal&>(*window));
    buffer<Vector3<Tfloat>> x(device, NUM_PARTICLES()); // OpenGL buffer
    buffer<Vector4<Tfloat>> color(device, NUM_PARTICLES());
#elif WITH_OPENGL
    opengl_buffer<Vector3<Tfloat>> x(device, NUM_PARTICLES()); // OpenGL buffer
    opengl_buffer<Vector4<Tfloat>> color(device, NUM_PARTICLES());
#else
    buffer<Vector3<Tfloat>> x(device, NUM_PARTICLES());
    buffer<Vector4<Tfloat>> color(device, NUM_PARTICLES());
#endif

    using T = Tfloat;

    size_t tree_size = NUM_PARTICLES() * TREE_FACTOR() + 100000;

    unique_ptr<cosmos_base<T>> Cosmos;

    Cosmos = make_unique<cosmos<T, MULTIPOLE_ORDER>>(device, NUM_PARTICLES(), tree_size, MAX_DISTFAC(), MAX_NODESIZE());

    if (argc >= 2)
    {
        Cosmos->make_IC(argv[1]);
    }

    // cosmos<T, MULTIPOLE_ORDER> Cosmos(device, NUM_PARTICLES(), MAX_DISTFAC());

    kernel set_colors(device, [&](const resource<Vector<T, 3>>& cx) {
        gpu_for_global(0, x.size(), [&](gpu_uint k) {
            color[k] = ::color(static_cast<gpu_float>(Cosmos->potential[k]));
            x[k] = cx[k].cast<gpu_float>();
            // Tweaking z coordinate to use potential for depth testing.
            // Particles are displayed according to their x and y coordinates.
            // If multiple particles are drawn at the same pixel, the one with the
            // highest potential will be shown.
            x[k][2] = static_cast<gpu_float>(-Cosmos->potential[k]) * 0.01f;
        });
    });

    auto last_fps_time = steady_clock::now();
    size_t last_fps_step = 0;

    bool quit = false;
    constexpr uint make_tree_every = 1; // doesn't work yet with > 1

    for (size_t step = 0; !quit; ++step)
    {
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

        Cosmos->movefunc(0.5f * DT());
        if (step % make_tree_every == 0)
        {
            Cosmos->make_tree();
        }
        Cosmos->compute_force();

        if (PRECISION_TEST() && step / make_tree_every % 128 == 0)
        {
            Cosmos->precision_test();
        }

        Cosmos->kick(DT());
        Cosmos->movefunc(0.5 * DT());

        auto now = steady_clock::now();
        if (now - last_fps_time > chrono::seconds(1))
        {
            stringstream title;
            Tdouble rate = (step - last_fps_step) / chrono::duration<double>(now - last_fps_time).count();
            title << "N-body. N=" << x.size() << ", step " << step << ", " << rate << " fps, device=" << device.name();
            window->set_title(title.str());
            last_fps_step = step;
            last_fps_time = now;
        }

        set_colors(Cosmos->x);

#if WITH_METAL
        Renderer.render(x);
#elif WITH_OPENGL
        render(window->window, x, &color);
        SDL_GL_SwapWindow(window->window);
#else
        cout << "x=" << x << endl;
#endif
    }
    return 0;
}
