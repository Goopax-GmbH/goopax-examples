/**
   \example gpudirect.cpp
   GPUDirect example program

   This program shows how GPUDirect *should* work with goopax, though it hasn't been well tested, yet.
   GPUDirect can increase the speed of data transfer from certain cameras.
   An Nvidia GPU with GPUDirect support is required, see
   https://developer.nvidia.com/gpudirectforvideo for a list of supported GPUs.
 */

#include <goopax>
#include <unistd.h>

using namespace goopax;
using namespace std;

int main()
{
    // Create Goopax device with CUDA backend
    goopax_device device = default_device(env_CUDA);

    // hard-coding image size for simplicity
    const int width = 640;
    const int height = 480;

#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const size_t pagesize = si.dwPageSize;
#else
    const size_t pagesize = getpagesize();
#endif

    // Allocate page-aligned host memory
    char* host_p = static_cast<char*>(operator new[]((width * height * sizeof(char) + pagesize - 1) & ~(pagesize - 1),
                                                     static_cast<std::align_val_t>(pagesize)));

    // Create host-mapped goopax buffer
    buffer<char> goopax_buffer(device, width * height, host_p);

    // Create kernel. This kernel computes the sum of all pixels.
    kernel get_sum(device, [&goopax_buffer]() {
        gpu_uint sum = 0;
        gpu_for_global(0, width * height, [&](gpu_uint k) { sum += goopax_buffer[k]; });
        return gather_add(sum);
    });

    // start of image loop...

    /*
      Transferring the image data. In this example program, we use std::fill for simplicity.
      When using real cameras, use the functions provided from the camera vendor, instead.

      If using basler cameras, this should probably work (along with other functions):

          size_t piSize = width*height*sizeof(char);
          GDMemcpy(host_p, &piSize);

    */
    fill(host_p, host_p + width * height, 17);

    // Do GPU work
    cout << "average: " << get_sum().get() / (width * height) << endl;

    // end of image loop

    operator delete[](host_p, static_cast<std::align_val_t>(pagesize));
}
