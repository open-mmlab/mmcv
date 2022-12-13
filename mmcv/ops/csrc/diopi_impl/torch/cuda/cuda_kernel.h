// Copyright (c) OpenMMLab. All rights reserved
#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H

#include <ATen/ATen.h>

namespace mmcv {
namespace diopiops {

using namespace at;

Tensor NMSCUDAKernelLauncher(Tensor boxes, Tensor scores, float iou_threshold,
                             int64_t offset);

void ChamferDistanceForwardCUDAKernelLauncher(
    const Tensor xyz1, const Tensor xyz2, const Tensor dist1,
    const Tensor dist2, const Tensor idx1, const Tensor idx2);

void ChamferDistanceBackwardCUDAKernelLauncher(
    const Tensor xyz1, const Tensor xyz2, Tensor idx1, Tensor idx2,
    Tensor grad_dist1, Tensor grad_dist2, Tensor grad_xyz1, Tensor grad_xyz2);

} // namespace diopiops
} // namespace mmcv
#endif  // CUDA_KERNEL_H
