//  Copyright © 2022 Apple Inc.

// This file is modify from:
// https://github.com/pytorch/pytorch/blob/a85d1f0bcdd02cf18d3b0517337458cb51a18cdb/aten/src/ATen/mps/MPSDevice.h

#pragma once
#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
typedef id<MTLDevice> MTLDevice_t;
#else
typedef void* MTLDevice;
typedef void* MTLDevice_t;
#endif

using namespace std;

namespace at {
namespace mps {

//-----------------------------------------------------------------
//  MPSDevice
//
// MPSDevice is a singleton class that returns the default device
//-----------------------------------------------------------------

class TORCH_API MPSDevice {
 public:
  /**
   * MPSDevice should not be cloneable.
   */
  MPSDevice(MPSDevice& other) = delete;
  /**
   * MPSDevice should not be assignable.
   */
  void operator=(const MPSDevice&) = delete;
  /**
   * Gets single instance of the Device.
   */
  static MPSDevice* getInstance();
  /**
   * Returns the single device.
   */
  MTLDevice_t device() { return _mtl_device; }

  ~MPSDevice();

 private:
  static MPSDevice* _device;
  MTLDevice_t _mtl_device;
  MPSDevice();
};

TORCH_API bool is_available();

TORCH_API at::Allocator* GetMPSAllocator(bool useSharedAllocator = false);

}  // namespace mps
}  // namespace at
