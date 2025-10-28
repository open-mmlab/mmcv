#ifndef PYTORCH_CUDA_HELPER
#define PYTORCH_CUDA_HELPER

#include <ATen/ATen.h>
#ifdef MMCV_WITH_MUSA
#include "common_musa_helper.hpp"
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/share/generated_cuda_compatible/aten/src/THC/THCAtomics.muh"
#include "torch_musa/share/generated_cuda_compatible/include/ATen/musa/MUSAApplyUtils.muh"
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

// ======================================================
// 判断 PyTorch 版本：
// 如果版本 >= 2.5.0，使用新的 MUSAApplyUtils.muh
// 否则，使用 MUSA_Port_ApplyUtils.muh
// ======================================================
#if defined(TORCH_VERSION_MAJOR) && defined(TORCH_VERSION_MINOR)
  #if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 5)
    #pragma message("[MUSA HELPER] Using torch_musa/share/generated_cuda_compatible/include/ATen/musa/MUSAApplyUtils.muh for PyTorch >= 2.5.0")
    #include "torch_musa/share/generated_cuda_compatible/include/ATen/musa/MUSAApplyUtils.muh"
  #else
    #pragma message("[MUSA HELPER] Using torch_musa/share/generated_cuda_compatible/include/ATen/musa/MUSA_Port_ApplyUtils.muh for PyTorch < 2.5.0")
    #include "torch_musa/share/generated_cuda_compatible/include/ATen/musa/MUSA_Port_ApplyUtils.muh"
  #endif
#else
  // 若未定义版本宏，默认使用新接口（兼容性考虑）
  #pragma message("[MUSA HELPER] TORCH_VERSION_MAJOR/MINOR not defined, defaulting to MUSAApplyUtils.muh")
  #include "torch_musa/share/generated_cuda_compatible/include/ATen/musa/MUSAApplyUtils.muh"
#endif

#endif  // PYTORCH_CUDA_HELPER
