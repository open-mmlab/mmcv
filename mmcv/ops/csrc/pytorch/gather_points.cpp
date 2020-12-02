#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void GatherPointsCUDAKernelLauncher(int b, int c, int n, int npoints,
                                    const Tensor points, const Tensor idx,
                                    Tensor out);

int gather_points_cuda(int b, int c, int n, int npoints, const Tensor points,
                       const Tensor idx, Tensor out) {
  GatherPointsCUDAKernelLauncher(b, c, n, npoints, points, idx, out);
}

void GatherPointsBackwardCUDAKernelLauncher(int b, int c, int n, int npoints,
                                            const Tensor grad_out,
                                            const Tensor idx,
                                            Tensor grad_points);

int gather_points_backward_cuda(int b, int c, int n, int npoints,
                                const Tensor grad_out, const Tensor idx,
                                Tensor grad_points) {
  GatherPointsBackwardCUDAKernelLauncher(b, c, n, npoints, grad_out, idx,
                                         grad_points);
}

#endif

int gather_points(int b, int c, int n, int npoints, Tensor points, Tensor idx,
                  Tensor out) {
  if (points.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(points);
    CHECK_CUDA_INPUT(idx);
    CHECK_CUDA_INPUT(out);

    gather_points_cuda(b, c, n, npoints, points, idx, out);
    return 1;
#else
    AT_ERROR("gather_points_backward is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("gather_points_backward is not implemented on CPU");
  }
}

int gather_points_backward(int b, int c, int n, int npoints, Tensor grad_out,
                           Tensor idx, Tensor grad_points) {
  if (grad_out.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(grad_out);
    CHECK_CUDA_INPUT(idx);
    CHECK_CUDA_INPUT(grad_points);

    gather_points_backward_cuda(b, c, n, npoints, grad_out, idx, grad_points);
    return 1;
#else
    AT_ERROR("gather_points_backward is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("gather_points_backward is not implemented on CPU");
  }
}
