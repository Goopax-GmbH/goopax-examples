/**
   \example pi.cpp
   Throwing darts to approximate the value of pi
 */

#include <chrono>
#include <goopax>
#include <goopax_extra/random.hpp>
#include <goopax_extra/struct_types.hpp>
#include <random>

using namespace goopax;
using namespace std;

static constexpr uint32_t N = 100000;
static constexpr unsigned int Nsub = 8;

int main()
{
    vector<kernel<uint64_t()>> kernels;

    std::random_device rd;
    for (goopax_device device : goopax::devices(env_ALL))
    {
        WELL512_data rnd(device, device.default_global_size_max(), rd());

        kernel newkernel(device, [&rnd]() -> gather_add<uint64_t> {
            WELL512_lib rndlib(rnd);

            gpu_uint num = 0;

            gpu_for(0, N, [&](gpu_int) {
                array<gpu_uint, Nsub * 2> rnd_values = rndlib.generate();
                for (unsigned int sub = 0; sub < Nsub; ++sub)
                {
                    // get x and y values in range 0..1
                    gpu_float x = rnd_values[2 * sub] * gpu_float(1.f / (uint64_t(1) << 32));
                    gpu_float y = rnd_values[2 * sub + 1] * gpu_float(1.f / (uint64_t(1) << 32));

                    num += (gpu_uint)(x * x + y * y < 1.f);
                }
            });
            return static_cast<gpu_uint64>(num);
        });
        cout << "Device " << kernels.size() << ": " << device.name() << ", #threads: " << newkernel.global_size()
             << ", envmode=" << device.get_envmode() << endl;

        kernels.push_back(newkernel);
    }
    cout << endl;

    for (unsigned int k = 0; k < 10; ++k)
    {
        cout << "Running..." << std::flush;
        auto time_start = chrono::steady_clock::now();

        vector<goopax_future<uint64_t>> results(kernels.size());
        for (int i = 0; i < kernels.size(); ++i)
        {
            (results[i] = kernels[i]()).set_callback([i]() { cout << i << std::flush; });
        }

        uint64_t darts = 0;
        uint64_t hits = 0;
        for (int i = 0; i < kernels.size(); ++i)
        {
            darts += uint64_t(N) * Nsub * kernels[i].global_size();
            hits += results[i].get();
        }
        double pi = (4 * static_cast<double>(hits)) / darts;
        double time = chrono::duration<double>(chrono::steady_clock::now() - time_start).count();
        cout << " hit " << hits << "/" << darts << " darts -> pi=" << pi << ", time=" << time << " seconds." << endl;
    }
}
