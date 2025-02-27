#ifndef PYTORCH_CUDA_HELPER
#define PYTORCH_CUDA_HELPER

#include <ATen/ATen.h>
#ifdef MMCV_WITH_MUSA
#include "common_musa_helper.hpp"
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/share/generated_cuda_compatible/aten/src/THC/THCAtomics.muh"
#include "torch_musa/share/generated_cuda_compatible/include/ATen/musa/MUSA_PORT_ApplyUtils.muh"
#else
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <THC/THCAtomics.cuh>

#include "common_cuda_helper.hpp"
#endif

using at::Half;
using at::Tensor;
using phalf = at::Half;

#define __PHALF(x) (x)
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#endif  // PYTORCH_CUDA_HELPER
