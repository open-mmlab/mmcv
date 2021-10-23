#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void GatherPointsForwardCUDAKernelLauncher(int b, int c, int n, int npoints,
                                           const Tensor points,
                                           const Tensor idx, Tensor out);

void gather_points_forward_cuda(int b, int c, int n, int npoints,
                                const Tensor points, const Tensor idx,
                                Tensor out) {
  GatherPointsForwardCUDAKernelLauncher(b, c, n, npoints, points, idx, out);
};

void GatherPointsBackwardCUDAKernelLauncher(int b, int c, int n, int npoints,
                                            const Tensor grad_out,
                                            const Tensor idx,
                                            Tensor grad_points);

void gather_points_backward_cuda(int b, int c, int n, int npoints,
                                 const Tensor grad_out, const Tensor idx,
                                 Tensor grad_points) {
  GatherPointsBackwardCUDAKernelLauncher(b, c, n, npoints, grad_out, idx,
                                         grad_points);
};
#endif

void gather_points_forward(Tensor points_tensor, Tensor idx_tensor,
                           Tensor out_tensor, int b, int c, int n,
                           int npoints) {
  if (points_tensor.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    gather_points_forward_cuda(b, c, n, npoints, points_tensor, idx_tensor,
                               out_tensor);
#else
    AT_ERROR("gather_points is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("gather_points is not implemented on CPU");
  }
}

void gather_points_backward(Tensor grad_out_tensor, Tensor idx_tensor,
                            Tensor grad_points_tensor, int b, int c, int n,
                            int npoints) {
  if (grad_out_tensor.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    gather_points_backward_cuda(b, c, n, npoints, grad_out_tensor, idx_tensor,
                                grad_points_tensor);
#else
    AT_ERROR("gather_points is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("gather_points is not implemented on CPU");
  }
}
