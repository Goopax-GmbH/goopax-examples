/**
   \example goopaxinfo.cpp
   Display goopax device information
 */

#include <goopax>
#include <goopax_extra/output.hpp>
#include <goopax_extra/types.hpp>

using namespace goopax;
using namespace std;

static constexpr const char* backend_names[] = { "CPU", "OpenCL", "CUDA", "Metal", "Vulkan" };

namespace std
{
template<typename T, size_t N>
std::ostream& operator<<(std::ostream& s, const std::array<T, N>& a)
{
    s << "(";
    for (size_t k = 0; k < N; ++k)
    {
        if (k != 0)
        {
            s << ", ";
        }
        s << a[k];
    }
    s << ")";
    return s;
}

std::ostream& operator<<(std::ostream& s, const std::type_info* ti)
{
    if (ti != nullptr)
    {
        s << goopax::pretty_typename(*ti);
    }
    else
    {
        s << "unknown type";
    }
    return s;
}
}

int main()
{
    for (unsigned int log_env = 0; log_env < 10; ++log_env)
    {
        goopax::envmode env = static_cast<goopax::envmode>(1 << log_env);
        vector<goopax_device> dvec = goopax::devices(env);
        if (!dvec.empty())
        {
            cout << "\nbackend: " << env;
            if (log_env < std::size(backend_names))
            {
                cout << " (" << backend_names[log_env] << ")";
            }
            cout << endl;
            for (size_t device_i = 0; device_i < dvec.size(); ++device_i)
            {
                auto& d = dvec[device_i];
                cout << "  device " << device_i << "/" << dvec.size() << ":" << endl
                     << "    name=" << d.name() << endl
                     << "    vendor=" << d.vendor() << endl
                     << "    local_size: default=" << d.default_local_size() << ", max=" << d.max_local_size() << endl
                     << "    default global_size: " << d.default_global_size_min() << ".."
                     << d.default_global_size_max() << endl
                     << "    global memory: " << d.global_memory_total() << endl
                     << "    cache_line=" << d.cache_line() << endl
                     << "    cache_size=" << d.cache_size() << endl
                     << "    registers: " << d.max_registers() << endl;
                cout << "    supported types:";
                const char* delim = " ";

                for (unsigned int t = 0; t < std::size(type_enum_raw_names); ++t)
                {
                    if (d.support_type(static_cast<type_enum_t>(t * num_type_modes)))
                    {
                        cout << delim << type_enum_raw_names[t];
                        delim = ", ";
                    }
                }
                cout << endl;

                cout << "    warp_matrix modes:" << endl;

                if (d.get_matrix_support_table())
                {
                    for (auto* mi = d.get_matrix_support_table(); mi; mi = mi->next)
                    {
                        cout << "      " << *mi << endl;
                    }
                }
                else
                {
                    cout << "      NONE" << endl;
                }
            }
        }
    }
}
