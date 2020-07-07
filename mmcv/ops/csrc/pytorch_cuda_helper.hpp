#ifndef PYTORCH_CUDA_HELPER
#define PYTORCH_CUDA_HELPER

#include <ATen/ATen.h>
#ifdef __NVCC__
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <THC/THCAtomics.cuh>
#endif
#ifdef __HIP_PLATFORM_HCC__
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>

#include <THH/THHAtomics.cuh>
#endif

#include "common_cuda_helper.hpp"

using at::Half;
using at::Tensor;
using phalf = at::Half;

#define __PHALF(x) (x)

#endif  // PYTORCH_CUDA_HELPER
