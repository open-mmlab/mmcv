#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void gather_points_cuda_forward(int b, int c, int n, int npoints,
                                const float *points, const int *idx,
                                float *out);

void gather_points_cuda_backward(int b, int c, int n, int npoints,
                                 const float *grad_out, const int *idx,
                                 float *grad_points);
#endif

int gather_points_forward(int b, int c, int n, int npoints,
                          at::Tensor points_tensor, at::Tensor idx_tensor,
                          at::Tensor out_tensor) {
  if (points_tensor.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    const float *points = points_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    float *out = out_tensor.data_ptr<float>();

    gather_points_cuda_forward(b, c, n, npoints, points, idx, out);
    return 1;
#else
    AT_ERROR("gather_points is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("gather_points is not implemented on CPU");
  }
}

int gather_points_backward(int b, int c, int n, int npoints,
                           at::Tensor grad_out_tensor, at::Tensor idx_tensor,
                           at::Tensor grad_points_tensor) {
  if (grad_out_tensor.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    const float *grad_out = grad_out_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    float *grad_points = grad_points_tensor.data_ptr<float>();

    gather_points_cuda_backward(b, c, n, npoints, grad_out, idx, grad_points);
    return 1;
#else
    AT_ERROR("gather_points is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("gather_points is not implemented on CPU");
  }
}
