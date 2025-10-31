/**
   \example fft.cpp
   Fast fourier transform example program.
   Applies fourier transforms to video or camera images to make them unsharp.
 */

#include <SDL3/SDL_main.h>
#include <goopax_draw/window_sdl.h>
#include <goopax_extra/fft.hpp>
#include <opencv2/opencv.hpp>

using namespace goopax;
using namespace std;
using Eigen::Vector;

struct fftdata
{
    static constexpr bool swap_RB = true;

    goopax_device device;
    Vector<Tuint, 2> size;
    buffer<Vector<uint8_t, 3>> inputbuf;
    buffer<complex<Tfloat>> tmp1;
    buffer<complex<Tfloat>> tmp2;

    array<kernel<void(const buffer<Vector<uint8_t, 3>>& input)>, 3> fft_x;
    kernel<void()> fft_y;
    kernel<void()> adjust_phase;
    kernel<void()> ifft_y;
    array<kernel<void(image_buffer<2, Vector<Tuint8_t, 4>, true>& frame)>, 3> ifft_x;

    void render(image_buffer<2, Vector<Tuint8_t, 4>, true>& drawimage)
    {
        for (unsigned int channel = 0; channel < 3; ++channel)
        {
            fft_x[channel](inputbuf);
            fft_y();
            adjust_phase();
            ifft_y();
            ifft_x[channel](drawimage);
        }
    }

    static void show_primes(unsigned int width)
    {
        Tuint k = 2;
        const char* separator = "";
        while (width != 1)
        {
            while (width % k != 0)
            {
                ++k;
            }
            cout << separator << k;
            separator = "*";
            width /= k;
            if (k >= 10)
            {
                cout << " [[Uh, that's a big prime factor. Performance may not be optimal.]] ";
            }
        }
    }

    fftdata(goopax_device device0, Vector<Tuint, 2> size0)
        : device(device0)
        , size(size0)
        , inputbuf(device, size[0] * size[1])
        , tmp1(device, size[0] * size[1])
        , tmp2(device, size[0] * size[1])

    {
        for (unsigned int channel = 0; channel < 3; ++channel)
        {
            fft_x[channel].assign(device, [this, channel](const resource<Vector<uint8_t, 3>>& input) {
                const uint ls = min(local_size(), ((size[0] ^ (size[0] - 1)) + 1) / 2);
                gpu_uint gid = global_id() / ls;
                const uint ng = global_size() / ls;
                cout << "fft_x: width=" << size[0] << " = ";
                show_primes(size[0]);
                cout << ", ls=" << ls << endl;

                gpu_for(gid, size[1], ng, [&](gpu_uint y) {
                    fft_workgroup<gpu_float>(
                        [&](gpu_uint x) {
                            return complex<gpu_float>(input[y * size[0] + x][channel] * (1.f / 255), 0);
                        },
                        [&](gpu_uint x, complex<gpu_float> value) { this->tmp1[y * size[0] + x] = value; },
                        size[0],
                        ls);
                });
            });
        }

        fft_y.assign(device, [this]() {
            const unsigned int ls = min(local_size(), ((size[1] ^ (size[1] - 1)) + 1) / 2);
            gpu_uint lid = global_id() % ls;
            gpu_uint gid = global_id() / ls;
            const unsigned int ng = global_size() / ls;

            cout << "fft_y: height=" << size[1] << " = ";
            show_primes(size[1]);
            cout << ", ls=" << ls << endl;

            gpu_for(gid, size[0], ng, [&](gpu_uint x) {
                fft_workgroup<gpu_float>(
                    [&](gpu_uint y) { return this->tmp1[y * size[0] + x]; },
                    [&](gpu_uint y, complex<gpu_float> value) { this->tmp2[y * size[0] + x] = value; },
                    size[1],
                    ls);
            });
        });
        ifft_y.assign(device, [this]() {
            const unsigned int ls = min(local_size(), ((size[1] ^ (size[1] - 1)) + 1) / 2);
            gpu_uint lid = global_id() % ls;
            gpu_uint gid = global_id() / ls;
            const unsigned int ng = global_size() / ls;
            gpu_for(gid, size[0], ng, [&](gpu_uint x) {
                ifft_workgroup<gpu_float>(
                    [&](gpu_uint y) { return this->tmp2[y * size[0] + x]; },
                    [&](gpu_uint y, complex<gpu_float> value) { this->tmp1[y * size[0] + x] = value; },
                    size[1],
                    ls);
            });
        });

        for (unsigned int channel = 0; channel < 3; ++channel)
        {
            ifft_x[channel].assign(device, [this, channel](image_resource<2, Vector<Tuint8_t, 4>, true>& frame) {
                const unsigned int ls = min(local_size(), ((size[0] ^ (size[0] - 1)) + 1) / 2);
                gpu_uint lid = global_id() % ls;
                gpu_uint gid = global_id() / ls;
                const unsigned int ng = global_size() / ls;

                gpu_for(gid, size[1], ng, [&](gpu_uint y) {
                    ifft_workgroup<gpu_float>([&](gpu_uint x) { return this->tmp1[y * size[0] + x]; },
                                              [&](gpu_uint x, complex<gpu_float> value) {
                                                  Vector<gpu_float, 4> c = frame.read({ x, y });
                                                  c[swap_RB ? 2 - channel : channel] = value.real();
                                                  if (channel == 2)
                                                  {
                                                      c[3] = 1;
                                                  }
                                                  frame.write({ x, y }, c);
                                              },
                                              size[0],
                                              ls);
                });
            });
        }

        adjust_phase.assign(device, [this]() {
            gpu_for_group(0, size[1], [&](gpu_uint y) {
                gpu_for_local(0, size[0], [&](gpu_uint x) {
                    // Doing some stuff with the FFT image. Suppressing short wavelengths.
                    this->tmp2[y * size[0] + x] *= 20.f / (20.f + x * x + y * y);
                });
            });
        });
    }
};

int main(int argc, char** argv)
{
    cv::VideoCapture cap; // open the video file for reading
    if (argc >= 2)
    {
        cout << "Reading video file " << argv[1] << endl;
        cap = cv::VideoCapture(argv[1]); // open the video file for reading
    }
    else
    {
        cap = cv::VideoCapture(0); // open the camera for reading
    }

    if (!cap.isOpened()) // if not success, exit program
    {
        cout << "Cannot open the video file" << endl;
        return (EXIT_FAILURE);
    }

    unique_ptr<fftdata> data;
    unique_ptr<sdl_window> window;

    bool quit = false;
    while (!quit)
    {
        cv::Mat frame;

        bool bSuccess = cap.read(frame); // read a new frame from video
        if (!bSuccess)
        {
            cout << "Cannot read the frame from video file" << endl;
            break;
        }

        if (data.get() == nullptr)
        {
            // Now that we have the image size, create a window and the GPU kernels.
            Vector<Tuint, 2> size = { frame.cols, frame.rows };
            window = sdl_window::create("fft", size);
            data = make_unique<fftdata>(window->device, size);
        }

        assert(frame.elemSize() == 3); // Assuming 24 bit RGB.

        window->draw_goopax([&](image_buffer<2, Vector<Tuint8_t, 4>, true>& image) {
            data->inputbuf.copy_from_host_async(reinterpret_cast<Vector<uint8_t, 3>*>(frame.data));
            data->render(image);
        });

        // Because there are no other synchronization points in this demo
        // (we are not evaluating any results from the GPU), this wait is
        // required to prevent endless submission of asynchronous kernel calls.
        window->device.wait_all();

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
                };
            }
        }
    }
}
