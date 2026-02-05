/**
   \example helloworld.cpp
   Greetings from every thread.
 */

#include <goopax>
using namespace goopax;

int main()
{
    std::vector<kernel<void()>> kernels;

    for (goopax_device device : devices())
    {
        kernels.emplace_back(device, []() { gpu_cout << "Hello from thread " << global_id() << std::endl; });
    }

    for (auto& hello : kernels)
    {
        hello();
    }
}
