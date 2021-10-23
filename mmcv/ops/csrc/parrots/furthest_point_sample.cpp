// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/sampling.cpp

#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void FurthestPointSamplingForwardCUDAKernelLauncher(int b, int n, int m,
                                                    const float *dataset,
                                                    float *temp, int *idxs);

void furthest_point_sampling_forward_cuda(int b, int n, int m,
                                          const float *dataset, float *temp,
                                          int *idxs) {
  FurthestPointSamplingForwardCUDAKernelLauncher(b, n, m, dataset, temp, idxs);
}

void FurthestPointSamplingWithDistForwardCUDAKernelLauncher(
    int b, int n, int m, const float *dataset, float *temp, int *idxs);

void furthest_point_sampling_with_dist_forward_cuda(int b, int n, int m,
                                                    const float *dataset,
                                                    float *temp, int *idxs) {
  FurthestPointSamplingWithDistForwardCUDAKernelLauncher(b, n, m, dataset, temp,
                                                         idxs);
}
#endif

void furthest_point_sampling_forward(Tensor points_tensor, Tensor temp_tensor,
                                     Tensor idx_tensor, int b, int n, int m) {
  if (points_tensor.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    const float *points = points_tensor.data_ptr<float>();
    float *temp = temp_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();
    furthest_point_sampling_forward_cuda(b, n, m, points, temp, idx);
#else
    AT_ERROR("furthest_point_sampling is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("furthest_point_sampling is not implemented on CPU");
  }
}

void furthest_point_sampling_with_dist_forward(Tensor points_tensor,
                                               Tensor temp_tensor,
                                               Tensor idx_tensor, int b, int n,
                                               int m) {
  if (points_tensor.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    const float *points = points_tensor.data<float>();
    float *temp = temp_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    furthest_point_sampling_with_dist_forward_cuda(b, n, m, points, temp, idx);
#else
    AT_ERROR(
        "furthest_point_sampling_with_dist is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("furthest_point_sampling_with_dist is not implemented on CPU");
  }
}
