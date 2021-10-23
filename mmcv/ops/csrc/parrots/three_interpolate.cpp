// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/interpolate.cpp

#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void ThreeInterpolateForwardCUDAKernelLauncher(int b, int c, int m, int n,
                                               const Tensor points,
                                               const Tensor idx,
                                               const Tensor weight, Tensor out);

void three_interpolate_forward_cuda(int b, int c, int m, int n,
                                    const Tensor points, const Tensor idx,
                                    const Tensor weight, Tensor out) {
  ThreeInterpolateForwardCUDAKernelLauncher(b, c, m, n, points, idx, weight,
                                            out);
};

void ThreeInterpolateBackwardCUDAKernelLauncher(int b, int c, int n, int m,
                                                const Tensor grad_out,
                                                const Tensor idx,
                                                const Tensor weight,
                                                Tensor grad_points);

void three_interpolate_backward_cuda(int b, int c, int n, int m,
                                     const Tensor grad_out, const Tensor idx,
                                     const Tensor weight, Tensor grad_points) {
  ThreeInterpolateBackwardCUDAKernelLauncher(b, c, n, m, grad_out, idx, weight,
                                             grad_points);
};
#endif

void three_interpolate_forward(Tensor points_tensor, Tensor idx_tensor,
                               Tensor weight_tensor, Tensor out_tensor, int b,
                               int c, int m, int n) {
  if (points_tensor.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    three_interpolate_forward_cuda(b, c, m, n, points_tensor, idx_tensor,
                                   weight_tensor, out_tensor);
#else
    AT_ERROR("three_interpolate is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("three_interpolate is not implemented on CPU");
  }
}

void three_interpolate_backward(Tensor grad_out_tensor, Tensor idx_tensor,
                                Tensor weight_tensor, Tensor grad_points_tensor,
                                int b, int c, int n, int m) {
  if (grad_out_tensor.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    three_interpolate_backward_cuda(b, c, n, m, grad_out_tensor, idx_tensor,
                                    weight_tensor, grad_points_tensor);
#else
    AT_ERROR("three_interpolate is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("three_interpolate is not implemented on CPU");
  }
}
