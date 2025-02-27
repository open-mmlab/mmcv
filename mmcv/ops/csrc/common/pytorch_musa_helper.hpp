#ifndef PYTORCH_MUSA_HELPER
#define PYTORCH_MUSA_HELPER

#include <ATen/ATen.h>

#include <ATen/musa/MUSA_PORT_ApplyUtils.muh>
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

#endif  // PYTORCH_CUDA_HELPER
