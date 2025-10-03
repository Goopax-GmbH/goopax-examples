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
    goopax_device device = default_device(env_ALL);

    kernel foo(device, [](resource<int>& A) {
        gpu_float x = 0;
        for_each_global(A.begin(), A.end(), [&](auto& a) {
            gpu_if(a > 4)
            {
                x += a;
            }
        });
        for_each_global(A.begin(), A.end(), [&](auto& a) {
            gpu_if(a > 5)
            {
                x += a;
            }
        });
        for_each_global(A.begin(), A.end(), [&](auto& a) {
            gpu_if(a > 6)
            {
                x += a;
            }
        });
        for_each_global(A.begin(), A.end(), [&](auto& a) { x += a; });
        for_each_global(A.begin(), A.end(), [&](auto& a) { a += x; });
    });

    buffer<int> A(device, 1000);
    foo(A);

    cout << "A=" << A << endl;
}
