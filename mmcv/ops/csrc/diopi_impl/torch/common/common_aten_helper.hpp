#ifndef COMMON_ATEN_HELPER
#define COMMON_ATEN_HELPER

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <THC/THCAtomics.cuh>

#include "common_cuda_helper.hpp"

namespace mmcv {
namespace diopiops {

using at::Half;
using at::Tensor;
using phalf = at::Half;

#define __PHALF(x) (x)
} // namespace diopiops
} // namespace mmcv
#endif  // COMMON_ATEN_HELPER