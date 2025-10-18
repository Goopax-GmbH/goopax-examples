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
    kernel foo(device, [](resource<int>& A) { gpu_for_global(0, A.size(), [&](gpu_uint i) { A[i] = i; }); });

    buffer<int> A(device, 1000);
    foo(A);

    cout << "A=" << A << endl;
}
