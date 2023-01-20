// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==----- subsubdevice_cslice.cpp - SYCL subsubdevice basic test -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This test tries to identify devices at all levels of hierarchy.
//===----------------------------------------------------------------------===//

#if 1 // __STANDALONE_TESTING__
// unnamed namespace for standalone testing
#include <cstdlib>
#include <cstring>
namespace {
    void readEnvStringWithDefault(const char* name, char* value, const char* def)
    {
        char* tmp = std::getenv(name);
        if (!tmp) std::strcpy(value, def);
        else std::strcpy(value, tmp);
    }
}
#else
void readEnvStringWithDefault(const char*, char*, const char*);
#endif

#if __clang_major__ > 15
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
inline namespace cl {
    using namespace sycl;
}
namespace sycl = ::cl::sycl;
#endif

#include<map>
#include<vector>
#include<atomic>
#include<thread>
#include<exception>

static auto exception_handler = [](sycl::exception_list list) {
  for (std::exception_ptr const& e : list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const& e) {
      std::cout << "Exception: " << e.what() << std::endl; std::fflush(stdout);
      throw;
      //std::terminate();
    }
  }
};

int identify_devices() {
  std::vector<sycl::device> devices;
  int verbose = 0;
  try {
    std::vector<sycl::device> all_devices = sycl::device::get_devices();
    char variable_name[255];
                
    readEnvStringWithDefault("XPU_USE_SUBDEVICES", variable_name, "1");
    auto use_subdevices = std::atoi(variable_name);
    int split_streams = 0;
    if (use_subdevices) {
      readEnvStringWithDefault("XPU_SPLIT_STREAMS", variable_name, "1");
      split_streams = std::atoi(variable_name);
    }

    readEnvStringWithDefault("XPU_VERBOSE", variable_name, "1");
    verbose = std::atoi(variable_name);

    readEnvStringWithDefault("XPU_DEVICE_NAME", variable_name, "Graphics");

    if (verbose) {
      std::cout << "XPU_DEVICE_NAME=" << variable_name << std::endl;
      std::cout << "Devices found:" << std::endl;
    }
    std::cout << "Number of root devices = " << all_devices.size() << std::endl;
    for (auto& device : all_devices) {
      if (verbose) {
        std::cout << "* Device: "
                  << device.get_info<sycl::info::device::name>()
                  << ", Backend: "
                  << device.get_platform().get_backend()
                  << std::endl;
      }
                    
      if (device.get_info<sycl::info::device::name>().find(variable_name) != std::string::npos) {
        // Select devices with the same backend only
        if (devices.empty() || (!devices.empty() &&
                                devices[0].get_platform().get_backend() == device.get_platform().get_backend())) {
          // Select subdevices if any
          auto device_partition_properties = device.get_info<sycl::info::device::partition_properties>();
          if (!use_subdevices || device_partition_properties.empty()) {
            devices.push_back(device);
          } else {
            for (int i = 0; i < device_partition_properties.size(); i++) {
              if (device_partition_properties[i] == sycl::info::partition_property::partition_by_affinity_domain) {
                auto subdevices = device.create_sub_devices<
                    sycl::info::partition_property::partition_by_affinity_domain>(
                        sycl::info::partition_affinity_domain::numa);
                std::cout << "Number of subdevices = " << subdevices.size() << "\n";
                for (int j = 0; j < subdevices.size(); j++) {
                  auto subdevice_partition_properties =
                      subdevices[j].get_info<sycl::info::device::partition_properties>();
                  if (!split_streams || subdevice_partition_properties.empty()) {
                    devices.push_back(subdevices[j]);
                  } else {
                    for (int i = 0; i < subdevice_partition_properties.size(); i++) {
                      if (subdevice_partition_properties[i] == sycl::info::partition_property::ext_intel_partition_by_cslice) {
			auto streams = subdevices[j].create_sub_devices<
                             sycl::info::partition_property::ext_intel_partition_by_cslice>();
                        std::cout << "Number of compute slices = " << streams.size() << "\n";
                        for (int j = 0; j < streams.size(); j++) {
                          devices.push_back(streams[j]);
                        }
                        break;
                      }
                    }
                  }
                }
                break;
              } else {
                devices.push_back(device);
              }
            }
          }
        }
      }
    }
    return 0;
  }
  catch (sycl::exception& e) {
    std::cout << "Sync sycl exception in initialize_queues(): " << e.what() << std::endl; std::fflush(stdout);
    return 1;
  }
}

int main() {
  return identify_devices();
}



