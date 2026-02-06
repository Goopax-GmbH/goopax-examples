/**
   \example simple.cpp
   Simple example program. A kernel writes numbers to a buffer, which are then printed out.
 */

#include <goopax>
#include <goopax_extra/output.hpp>

using namespace goopax;
using namespace std;

int main()
{
    goopax_device device = default_device(env_CUDA);

    const int width = 640;
    const int height = 480;

#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const size_t pagesize = si.dwPageSize;
#else
    const size_t pagesize = getpagesize();
#endif

    char* host_p =
        static_cast<char*>(operator new[](width * height * sizeof(char), static_cast<std::align_val_t>(pagesize)));

    buffer<char> goopax_buffer(device, width * height, host_p);

    kernel get_sum(device, [&goopax_buffer]() {
        gpu_uint sum = 0;
        gpu_for_global(0, width * height, [&](gpu_uint k) { sum += goopax_buffer[k]; });
        return gather_add(sum);
    });

    fill(host_p, host_p + width * height, 17);

    cout << "average: " << get_sum().get() / (width * height) << endl;

    operator delete[](host_p, static_cast<std::align_val_t>(pagesize));
}
