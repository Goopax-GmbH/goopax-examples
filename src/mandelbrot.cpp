/**
   \example mandelbrot.cpp
   Mandelbrot example

   Use mouse click/wheel or finger gestures to navigate

   Keys:
   - escape: quit
 */

#include <SDL3/SDL_main.h>
#include <chrono>
#include <goopax_draw/window_sdl.h>
#include <goopax_extra/struct_types.hpp>

using namespace goopax;
using namespace std;
using Eigen::Vector;
using goopax::detail::PI;
using std::chrono::steady_clock;

template<class D, typename window_size_t = unsigned int>
complex<D> calc_c(complex<D> center, D scale, Vector<D, 2> position, window_size_t window_size)
{
    // Calculate the value c for a given image point.
    // This function is called both from within the kernel and from the main function
    return center
           + static_cast<D>(scale / window_size[0])
                 * complex<D>(position[0] - window_size[0] * 0.5f, position[1] - window_size[1] * 0.5f);
}

static const array<complex<double>, 2> max_allowed_range = { { { -2, -2 }, { 2, 2 } } };
static complex<double> clamp_range(const complex<double>& x)
{
    return { std::clamp(x.real(), max_allowed_range[0].real(), max_allowed_range[1].real()),
             std::clamp(x.imag(), max_allowed_range[0].imag(), max_allowed_range[1].imag()) };
}

int main(int, char**)
{
    auto window = sdl_window::create("mandelbrot", { 640, 480 }, SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);

    kernel render(window->device,
                  [](image_resource<2, Vector<Tuint8_t, 4>, true>& image,
                     const complex<gpu_float> center,
                     const gpu_float scale) {
                      gpu_for_global(0,
                                     image.width() * image.height(),
                                     [&](gpu_uint k) // Parallel loop over all image points
                                     {
                                         Vector<gpu_uint, 2> position = { k % image.width(), k / image.width() };

                                         complex<gpu_float> c = calc_c(
                                             center, scale, position.cast<gpu_float>().eval(), image.dimensions());
                                         complex<gpu_float> z(0, 0);

                                         static constexpr unsigned int max_iter = 4096;
                                         gpu_uint iter = 0;

                                         // As soon as norm(z) >= 4, z is destined to diverge to inf. Delaying
                                         // until 10.f to make colors a bit smoother.
                                         gpu_while(iter < max_iter && norm(z) < 10.f)
                                         {
                                             for (int k = 0; k < 4; ++k) // Inner loop to speed up things.
                                             {
                                                 z = z * z + c; // The core formula of the mandelbrot set
                                                 ++iter;
                                             }
                                         }

                                         Vector<gpu_float, 4> color = { 0, 0, 0.4, 1 };

                                         gpu_if(norm(z) >= 4.f)
                                         {
                                             gpu_float x = (iter - log2(log2(norm(z)))) * 0.03f;
                                             color[0] = 0.5f + 0.5f * sinpi(x);
                                             color[1] = 0.5f + 0.5f * sinpi(x + static_cast<float>(2. / 3));
                                             color[2] = 0.5f + 0.5f * sinpi(x + static_cast<float>(4. / 3));
                                         }

                                         image.write(position, color); // Set the color according to the escape time
                                     });
                  });

    complex<double> moveto = { -0.796570904132624102, 0.183652206054726097 };

    double scale = 2.0;

    complex<double> center = moveto;
    double speed_zoom = 1E-2;
    constexpr float timescale = 1; // [in seconds]

    bool quit = false;

    auto last_draw_time = steady_clock::now();
    auto last_fps_time = last_draw_time;
    unsigned int framecount = 0;

    map<SDL_FingerID, Vector<float, 2>> finger_positions;

    auto get_finger_cm = [&]() -> Vector<float, 2> {
        Vector<float, 2> ret = { 0, 0 };
        for (auto& f : finger_positions)
        {
            ret += f.second;
        }
        return ret / finger_positions.size();
    };

    Vector<float, 2> last_finger_cm;
    Tfloat last_finger_distance;

    while (!quit)
    {
        std::array<unsigned int, 2> window_size = window->get_size();
        bool finger_change = false;
        while (auto e = window->get_event())
        {
            if (e->type == SDL_EVENT_QUIT)
            {
                quit = true;
            }
            else if (e->type == SDL_EVENT_FINGER_DOWN)
            {
                finger_positions.insert({ e->tfinger.fingerID, { e->tfinger.x, e->tfinger.y } });
                finger_change = true;
            }
            else if (e->type == SDL_EVENT_FINGER_UP)
            {
                finger_positions.erase(e->tfinger.fingerID);
                finger_change = true;
            }

            else if (e->type == SDL_EVENT_FINGER_MOTION)
            {
                finger_positions[e->tfinger.fingerID] = { e->tfinger.x, e->tfinger.y };
            }
            else if (e->type == SDL_EVENT_MOUSE_BUTTON_DOWN)
            {
                float x = 0, y = 0;
                SDL_GetMouseState(&x, &y);

                moveto = calc_c(center,
                                scale, // Set new center
                                { x, y },
                                window_size);
                moveto = clamp_range(moveto);
            }
            else if (e->type == SDL_EVENT_MOUSE_WHEEL)
            {
                speed_zoom -= e->wheel.y;
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
                        break;
                };
            }
        }

        if (!finger_positions.empty())
        {
            if (finger_positions.size() == 2)
            {
                // 2 finger zoom
                float finger_distance =
                    (finger_positions.begin()->second - next(finger_positions.begin())->second).norm();
                if (!finger_change)
                {
                    float factor = finger_distance / last_finger_distance;
                    speed_zoom -= log(factor);
                }
                last_finger_distance = finger_distance;
            }

            Vector<float, 2> finger_cm = get_finger_cm();
            if (!finger_change)
            {
                // dragging with one or multiple fingers

                const complex<double> shift =
                    calc_c(center, scale, { finger_cm[0] * window_size[0], finger_cm[1] * window_size[1] }, window_size)
                    - calc_c(center,
                             scale,
                             { last_finger_cm[0] * window_size[0], last_finger_cm[1] * window_size[1] },
                             window_size);

                center -= shift;
                moveto -= shift;
                center = clamp_range(center);
                moveto = clamp_range(moveto);
            }
            last_finger_cm = finger_cm;
        }

        auto now = steady_clock::now();
        auto dt = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_draw_time).count();

        center += (moveto - center) * (dt * (1.0 / timescale + abs(speed_zoom)));
        scale *= exp(speed_zoom * dt);
        speed_zoom *= exp(-dt / timescale);

        std::array<unsigned int, 2> render_size;

        window->draw_goopax([&](image_buffer<2, Vector<Tuint8_t, 4>, true>& image) {
            render(image, static_cast<complex<float>>(center), scale);
            render_size = image.dimensions();
        });

        // Because there are no other synchronization points in this demo
        // (we are not evaluating any results from the GPU), this wait is
        // required to prevent endless submission of asynchronous kernel calls.
        window->device.wait_all();

        ++framecount;
        if (now - last_fps_time > std::chrono::seconds(1))
        {
            stringstream title;
            auto rate = framecount / std::chrono::duration<double>(now - last_fps_time).count();
            title << "Mandelbrot: screen size=" << render_size[0] << "x" << render_size[1] << ", " << rate << " fps"
                  << ", device=" << window->device.name();
            window->set_title(title.str());
            framecount = 0;
            last_fps_time = now;
        }

        last_draw_time = now;
    }
    return 0;
}
