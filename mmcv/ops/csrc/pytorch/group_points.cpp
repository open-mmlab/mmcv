// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/group_points.cpp
#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void GroupPointsCUDAKernelLauncher(int b, int c, int n, int npoints,
                                   int nsample, const Tensor points,
                                   const Tensor idx, Tensor out);

void group_points_cuda(int b, int c, int n, int npoints, int nsample,
                       const Tensor points, const Tensor idx, Tensor out) {
  GroupPointsCUDAKernelLauncher(b, c, n, npoints, nsample, points, idx, out);
}

void GroupPointsBackwardCUDAKernelLauncher(int b, int c, int n, int npoints,
                                           int nsample, const Tensor grad_out,
                                           const Tensor idx,
                                           Tensor grad_points);

void group_points_backward_cuda(int b, int c, int n, int npoints, int nsample,
                                const Tensor grad_out, const Tensor idx,
                                Tensor grad_points) {
  GroupPointsBackwardCUDAKernelLauncher(b, c, n, npoints, nsample, grad_out,
                                        idx, grad_points);
}
#endif

int group_points_backward(int b, int c, int n, int npoints, int nsample,
                          Tensor grad_out, Tensor idx, Tensor grad_points) {
  if (grad_out.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(grad_out);
    CHECK_CUDA_INPUT(idx);
    CHECK_CUDA_INPUT(grad_points);

    group_points_backward_cuda(b, c, n, npoints, nsample, grad_out, idx,
                               grad_points);
    return 1;
#else
    AT_ERROR("group_points_backward is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("group_points_backward is not implemented on CPU");
  }
}

int group_points(int b, int c, int n, int npoints, int nsample, Tensor points,
                 Tensor idx, Tensor out) {
  if (points.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(points);
    CHECK_CUDA_INPUT(idx);
    CHECK_CUDA_INPUT(out);

    group_points_cuda(b, c, n, npoints, nsample, points, idx, out);
    return 1;
#else
    AT_ERROR("group_points is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("group_points is not implemented on CPU");
  }
}
