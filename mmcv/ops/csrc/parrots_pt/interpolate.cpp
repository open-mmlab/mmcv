// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/interpolate.cpp
#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void ThreeNNCUDAKernelLauncher(int b, int n, int m, const Tensor unknown,
                               const Tensor known, Tensor dist2, Tensor idx);

void three_nn_cuda(int b, int n, int m, const Tensor unknown,
                   const Tensor known, Tensor dist2, Tensor idx) {
  ThreeNNCUDAKernelLauncher(b, n, m, unknown, known, dist2, idx);
}

void ThreeInterpolateCUDAKernelLauncher(int b, int c, int m, int n,
                                        const Tensor points, const Tensor idx,
                                        const Tensor weight, Tensor out);

void three_interpolate_cuda(int b, int c, int m, int n, const Tensor points,
                            const Tensor idx, const Tensor weight, Tensor out) {
  ThreeInterpolateCUDAKernelLauncher(b, c, m, n, points, idx, weight, out);
}

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
}

#endif

void three_nn(int b, int n, int m, const Tensor unknown, const Tensor known,
              Tensor dist2, Tensor idx) {
  if (unknown.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(unknown);
    CHECK_CUDA_INPUT(known);
    CHECK_CUDA_INPUT(dist2);
    CHECK_CUDA_INPUT(idx);

    three_nn_cuda(b, n, m, unknown, known, dist2, idx);
#else
    AT_ERROR("three_nn is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("three_nn is not implemented on CPU");
  }
}

void three_interpolate(int b, int c, int m, int n, const Tensor points,
                       const Tensor idx, const Tensor weight, Tensor out) {
  if (points.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(points);
    CHECK_CUDA_INPUT(idx);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(out);

    three_interpolate_cuda(b, c, m, n, points, idx, weight, out);
#else
    AT_ERROR("three_interpolate is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("three_interpolate is not implemented on CPU");
  }
}

void three_interpolate_backward(int b, int c, int n, int m,
                                const Tensor grad_out, const Tensor idx,
                                const Tensor weight, Tensor grad_points) {
  if (grad_out.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(grad_out);
    CHECK_CUDA_INPUT(idx);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(grad_points);

    three_interpolate_backward_cuda(b, c, n, m, grad_out, idx, weight,
                                    grad_points);
#else
    AT_ERROR("three_interpolate_backward is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("three_interpolate_backward is not implemented on CPU");
  }
}
