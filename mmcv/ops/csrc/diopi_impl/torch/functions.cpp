
#include <diopi/functions.h>
#include <torch/nn.h>
#include <torch/optim.h>
#include <iostream>
#include <math.h>

#include "helper.hpp"
#include "cuda/cuda_kernel.h"

#define FLT_MIN		__FLT_MIN__

extern "C" {

diopiError_t diopiNms(diopiContextHandle_t ctx, diopiTensorHandle_t* out, const diopiTensorHandle_t dets,
        const diopiTensorHandle_t scores, double iouThreshold, int64_t offset) {
    auto atDets = impl::aten::buildATen(dets);
    auto atScores = impl::aten::buildATen(scores);
    auto atOut = mmcv::diopiops::NMSCUDAKernelLauncher(atDets, atScores, iouThreshold, offset);
    impl::aten::buildDiopiTensor(ctx, atOut, out);
}

diopiError_t diopiChamferDistance(diopiContextHandle_t ctx, const diopiTensorHandle_t xyz1,
                                  const diopiTensorHandle_t xyz2, diopiTensorHandle_t dist1, diopiTensorHandle_t dist2,
                                  diopiTensorHandle_t idx1, diopiTensorHandle_t idx2) {
    auto xyz1_in = impl::aten::buildATen(xyz1);
    auto xyz2_in = impl::aten::buildATen(xyz2);
    auto dist1_out = impl::aten::buildATen(dist1);
    auto dist2_out = impl::aten::buildATen(dist2);
    auto idx1_out = impl::aten::buildATen(idx1);
    auto idx2_out = impl::aten::buildATen(idx2);
    mmcv::diopiops::ChamferDistanceForwardCUDAKernelLauncher(
        xyz1_in, xyz2_in, dist1_out, dist2_out, idx1_out, idx2_out);
}

diopiError_t diopiChamferDistanceBackward(diopiContextHandle_t ctx, const diopiTensorHandle_t xyz1, const diopiTensorHandle_t xyz2,
                                            const diopiTensorHandle_t idx1, const diopiTensorHandle_t idx2, const diopiTensorHandle_t grad_dist1, const diopiTensorHandle_t grad_dist2,
                                            diopiTensorHandle_t grad_xyz1, diopiTensorHandle_t grad_xyz2) {
    auto xyz1_in = impl::aten::buildATen(xyz1);
    auto xyz2_in = impl::aten::buildATen(xyz2);
    auto idx1_in = impl::aten::buildATen(idx1);
    auto idx2_in = impl::aten::buildATen(idx2);
    auto grad_dist1_in = impl::aten::buildATen(grad_dist1);
    auto grad_dist2_in = impl::aten::buildATen(grad_dist2);
    auto grad_xyz1_out = impl::aten::buildATen(grad_xyz1);
    auto grad_xyz2_out = impl::aten::buildATen(grad_xyz2);
    mmcv::diopiops::ChamferDistanceBackwardCUDAKernelLauncher(
        xyz1_in, xyz2_in, idx1_in, idx2_in, grad_dist1_in, grad_dist2_in, grad_xyz1_out, grad_xyz2_out);
}

}  // extern "C"
