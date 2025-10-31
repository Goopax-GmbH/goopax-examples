/**
   \example svm-pingpong.cpp
   Multiple devices exchange messages during the runtime of the kernel.
 */

#include <atomic>
#include <chrono>
#include <goopax>
#include <goopax_extra/param.hpp>
#include <thread>

#if __cpp_lib_atomic_ref >= 201806
#define HAVE_ATOMIC_REF 1
#else
#define HAVE_ATOMIC_REF 0
#endif

using namespace goopax;
using namespace std::chrono;
using std::cout;
using std::endl;

PARAMOPT<bool> VERB("verb", false);

template<typename T>
void pingpong()
{
    using gpu_T = typename make_gpu<T>::type;

    std::vector<goopax_device> devices;

    for (auto& device : goopax::devices(env_ALL))
    {
        if (device.support_svm() && device.support_svm_atomics() && device.support_type<T>())
        {
            cout << "player " << devices.size();
            devices.push_back(device);
        }
        else
        {
            cout << "Ignoring device";
        }
        cout << ": " << device.name() << ", support_svm: " << device.support_svm()
             << ", svm atomics: " << device.support_svm_atomics() << ", support_type<"
             << goopax::pretty_typename(typeid(T)) << ">: " << device.support_type<T>() << endl;
    }

    // This `kernels` vector contains both the GPU kernels (implicitly converted to std::function),
    // and the host code function.
    std::vector<std::function<void(T*, T*, unsigned int, T)>> kernels;

    for (unsigned int k = 0; k < devices.size(); ++k)
    {
        kernels.emplace_back(
            kernel(devices[k], [my_id = k](gpu_type<T*> ptr, gpu_type<T*> data, gpu_uint number_of_players, gpu_T N) {
                gpu_if(global_id() == 0)
                {
                    gpu_for(my_id, N, number_of_players, [&](gpu_T expect) {
                        //  Waiting for our turn.
                        gpu_while(atomic_load(*ptr, std::memory_order_acquire, memory::system) != expect)
                        {
                        }

                        // Now it's our turn. Do some work.
                        data[0] += 1;

                        if (VERB)
                        {
                            gpu_ostream DUMP(cout);
                            DUMP << "kernel " << my_id << ". device=" << get_current_build_device().name()
                                 << ". sync: " << expect << " -> " << expect + 1 << endl;
                        }

                        // Handing over to the next id.
                        atomic_store(*ptr, expect + 1, std::memory_order_release, memory::system);
                    });
                }
            }));
    }

#if HAVE_ATOMIC_REF
    cout << "player " << devices.size() << ": host program" << endl;
    kernels.push_back([my_id = devices.size()](T* ptr, T* data, unsigned int number_of_players, T N) {
        std::atomic_ref<T> ref(*ptr);
        for (unsigned int expect = my_id; expect < N; expect += number_of_players)
        {
            //  Waiting for our turn.
            while (ref.load(std::memory_order_acquire) != expect)
            {
            }

            // Now it's our turn. Do some work.
            data[0] += 1;

            if (VERB)
            {
                cout << "host code. sync: " << expect << " -> " << expect + 1 << endl;
            }

            // Handing over to the next id.
            ref.store(expect + 1, std::memory_order_release);
        }
    });
#endif

    if (devices.size() == 0)
    {
        cout << "No suitable device found. Leaving." << endl;
        return;
    }
    if (kernels.size() == 1)
    {
        cout << "Only one device present. This is rather trivial and should succeed." << endl;
    }
    cout << endl;

    svm_buffer<T> svmbuf(devices[0], 1);
    svm_buffer<T> data(devices[0], 1);

    for (unsigned int N = 1; N != 0; N <<= 1)
    {
        svmbuf[0] = 0;
        data[0] = 0;

        auto t0 = steady_clock::now();

        for (auto& k : kernels)
        {
            k(svmbuf.data(), data.data(), kernels.size(), N);
        }

        wait_all_devices();
        auto dt = duration<double>(steady_clock::now() - t0);
        cout << "N=" << N << ", time=" << dt.count() << ", average ping time: " << dt.count() * 1E6 / N
             << " microseconds"
             << ", data=" << data[0] << endl;
        assert(data[0] == N);

        if (dt > 1000ms)
            break;
    }
}

int main(int argc, char** argv)
{
    cout << "svm pingpong." << endl
         << "Multiple devices talk to each other." << endl
         << "This requires atomic operations on svm memory, and may not work on all graphics cards." << endl
         << endl;

    init_params(argc, argv);

#if GOOPAX_DEBUG
    pingpong<debugtype<unsigned int>>();
#else
    pingpong<unsigned int>();
#endif
}
