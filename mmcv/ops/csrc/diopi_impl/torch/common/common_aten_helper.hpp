#ifndef DIOPI_IMPL_TORCH_COMMON_ATEN_HELPER
#define DIOPI_IMPL_TORCH_COMMON_ATEN_HELPER

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
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

} // namespace diopiops
} // namespace mmcv
#endif  // DIOPI_IMPL_TORCH_COMMON_ATEN_HELPER