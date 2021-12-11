// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/group_points.cpp

#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void GroupPointsForwardCUDAKernelLauncher(int b, int c, int n, int npoints,
                                          int nsample, const Tensor points,
                                          const Tensor idx, Tensor out);
void group_points_forward_cuda(int b, int c, int n, int npoints, int nsample,
                               const Tensor points, const Tensor idx,
                               Tensor out) {
  GroupPointsForwardCUDAKernelLauncher(b, c, n, npoints, nsample, points, idx,
                                       out);
};

void GroupPointsBackwardCUDAKernelLauncher(int b, int c, int n, int npoints,
                                           int nsample, const Tensor grad_out,
                                           const Tensor idx,
                                           Tensor grad_points);
void group_points_backward_cuda(int b, int c, int n, int npoints, int nsample,
                                const Tensor grad_out, const Tensor idx,
                                Tensor grad_points) {
  GroupPointsBackwardCUDAKernelLauncher(b, c, n, npoints, nsample, grad_out,
                                        idx, grad_points);
};
#endif

void group_points_forward(Tensor points_tensor, Tensor idx_tensor,
                          Tensor out_tensor, int b, int c, int n, int npoints,
                          int nsample) {
  if (points_tensor.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    group_points_forward_cuda(b, c, n, npoints, nsample, points_tensor,
                              idx_tensor, out_tensor);
#else
    AT_ERROR("group_points is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("group_points is not implemented on CPU");
  }
}

void group_points_backward(Tensor grad_out_tensor, Tensor idx_tensor,
                           Tensor grad_points_tensor, int b, int c, int n,
                           int npoints, int nsample) {
  if (grad_out_tensor.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    group_points_backward_cuda(b, c, n, npoints, nsample, grad_out_tensor,
                               idx_tensor, grad_points_tensor);
#else
    AT_ERROR("group_points is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("group_points is not implemented on CPU");
  }
}
