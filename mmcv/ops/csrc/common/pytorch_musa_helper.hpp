#ifndef PYTORCH_MUSA_HELPER
#define PYTORCH_MUSA_HELPER

#include <ATen/ATen.h>

#include <THC/THCAtomics.muh>

#include "common_musa_helper.hpp"
#include "torch_musa/csrc/aten/musa/Exceptions.h"
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

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
    #pragma message("[MUSA HELPER] Using <ATen/musa/MUSAApplyUtils.muh> for PyTorch >= 2.5.0")
    #include <ATen/musa/MUSAApplyUtils.muh>
  #else
    #pragma message("[MUSA HELPER] Using <ATen/musa/MUSA_Port_ApplyUtils.muh> for PyTorch < 2.5.0")
    #include <ATen/musa/MUSA_Port_ApplyUtils.muh>
  #endif
#else
  // 若未定义版本宏，默认使用新接口（兼容性考虑）
  #pragma message("[MUSA HELPER] TORCH_VERSION_MAJOR/MINOR not defined, defaulting to MUSAApplyUtils.muh")
  #include <ATen/musa/MUSAApplyUtils.muh>
#endif

#endif  // PYTORCH_CUDA_HELPER


