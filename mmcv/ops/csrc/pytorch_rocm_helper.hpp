#ifndef PYTORCH_CUDA_HELPER
#define PYTORCH_CUDA_HELPER

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>

#include <THH/THHAtomics.cuh>

#include "common_rocm_helper.hpp"

using at::Half;
using at::Tensor;
using phalf = at::Half;

#define __PHALF(x) (x)

#endif  // PYTORCH_CUDA_HELPER
