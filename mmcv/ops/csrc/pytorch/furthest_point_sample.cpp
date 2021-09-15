// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/sampling.cpp

#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void furthest_point_sampling_kernel_launcher(int b, int n, int m,
                                             const float *dataset, float *temp,
                                             int *idxs);

void furthest_point_sampling_with_dist_kernel_launcher(int b, int n, int m,
                                                       const float *dataset,
                                                       float *temp, int *idxs);
#endif

int furthest_point_sampling_forward(int b, int n, int m,
                                    at::Tensor points_tensor,
                                    at::Tensor temp_tensor,
                                    at::Tensor idx_tensor) {
  if (points_tensor.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    const float *points = points_tensor.data_ptr<float>();
    float *temp = temp_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();

    furthest_point_sampling_kernel_launcher(b, n, m, points, temp, idx);
    return 1;
#else
    AT_ERROR("furthest_point_sampling is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("furthest_point_sampling is not implemented on CPU");
  }
}

int furthest_point_sampling_with_dist_forward(int b, int n, int m,
                                              at::Tensor points_tensor,
                                              at::Tensor temp_tensor,
                                              at::Tensor idx_tensor) {
  if (points_tensor.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    const float *points = points_tensor.data<float>();
    float *temp = temp_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    furthest_point_sampling_with_dist_kernel_launcher(b, n, m, points, temp,
                                                      idx);
    return 1;
#else
    AT_ERROR(
        "furthest_point_sampling_with_dist is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("furthest_point_sampling_with_dist is not implemented on CPU");
  }
}
