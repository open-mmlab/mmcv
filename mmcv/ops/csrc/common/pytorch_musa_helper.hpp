#ifndef PYTORCH_MUSA_HELPER
#define PYTORCH_MUSA_HELPER

#include <ATen/ATen.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>

// #include <ATen/cuda/CUDAApplyUtils.cuh>
// #include <THC/THCAtomics.cuh>

#include "common_musa_helper.hpp"

using at::Half;
using at::Tensor;
using phalf = at::Half;

#define __PHALF(x) (x)
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#endif  // PYTORCH_CUDA_HELPER
