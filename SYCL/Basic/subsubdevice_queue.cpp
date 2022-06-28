// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==------------ subdevice.cpp - SYCL subdevice basic test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <utility>

using namespace cl::sycl;

int main() {
  try {
    auto devices = device::get_devices();
    std::vector<device> SubSubDevicesDomainNuma;
    for (const auto &dev : devices) {
      // TODO: implement subdevices creation for host device
      if (dev.is_host())
        continue;

      assert(dev.get_info<info::device::partition_type_property>() ==
             info::partition_property::no_partition);

      size_t MaxSubDevices =
          dev.get_info<info::device::partition_max_sub_devices>();

      if (MaxSubDevices == 0)
        continue;

      try {
        auto SubDevicesDomainNuma = dev.create_sub_devices<
            info::partition_property::partition_by_affinity_domain>(
            info::partition_affinity_domain::numa);
        std::cout
            << "Created " << SubDevicesDomainNuma.size()
            << " subdevices using partition by numa affinity domain scheme."
            << std::endl;

        SubSubDevicesDomainNuma =
            SubDevicesDomainNuma[0]
                .create_sub_devices<
                    info::partition_property::partition_by_affinity_domain>(
                    info::partition_affinity_domain::numa);

        std::cout << "Created " << SubSubDevicesDomainNuma.size()
                  << " sub-subdevices from subdevice 0 using partition by numa "
                     "affinity domain scheme."
                  << std::endl;
      } catch (feature_not_supported) {
        // okay skip it
      }
    }
    if (!SubSubDevicesDomainNuma.empty()) {
      sycl::context context(SubSubDevicesDomainNuma);
      auto queue = sycl::queue{context, SubSubDevicesDomainNuma[0]};
      std::cout << "Created SubSubDevicesDomainNuma queue" << std::endl;

      float Data = 1.0;
      float Scalar = 2.0;
      sycl::buffer<float, 1> Buf(&Data, sycl::range<1>(1));
      auto Event = queue.submit([&](sycl::handler &cgh) {
        auto Acc = Buf.get_access<sycl::access::mode::read_write>(cgh);
        cgh.single_task<class test_event>([=]() { Acc[0] = Scalar; });
      });
      Event.wait();
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
