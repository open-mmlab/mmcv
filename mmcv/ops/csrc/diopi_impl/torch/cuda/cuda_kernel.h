// Copyright (c) OpenMMLab. All rights reserved
#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H

#include <ATen/ATen.h>

namespace mmcv {
namespace diopiops {

using namespace at;

Tensor NMSCUDAKernelLauncher(Tensor boxes, Tensor scores, float iou_threshold,
                             int64_t offset);

} // namespace diopiops
} // namespace mmcv
#endif  // CUDA_KERNEL_H