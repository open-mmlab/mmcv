// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/sampling.cpp
#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void FurthestPointSamplingCUDAKernelLauncher(int b, int n, int m,
                                             const Tensor dataset_tensor,
                                             Tensor temp_tensor,
                                             Tensor idxs_tensor);

void furthest_point_sampling_cuda(int b, int n, int m, const Tensor points,
                                  Tensor temp, Tensor idx) {
  FurthestPointSamplingCUDAKernelLauncher(b, n, m, points, temp, idx);
}

void FurthestPointSamplingWithDistCUDAKernelLauncher(
    int b, int n, int m, const Tensor dataset_tensor, Tensor temp_tensor,
    Tensor idxs_tensor);

void furthest_point_sampling_with_dist_cuda(int b, int n, int m,
                                            const Tensor points, Tensor temp,
                                            Tensor idx) {
  FurthestPointSamplingWithDistCUDAKernelLauncher(b, n, m, points, temp, idx);
}

#endif

int furthest_point_sampling(int b, int n, int m, const Tensor points,
                            Tensor temp, Tensor idx) {
  if (points.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(points);
    CHECK_CUDA_INPUT(temp);
    CHECK_CUDA_INPUT(idx);

    furthest_point_sampling_cuda(b, n, m, points, temp, idx);
    return 1;
#else
    AT_ERROR("furthest_point_sampling is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("furthest_point_sampling is not implemented on CPU");
  }
}

int furthest_point_sampling_with_dist(int b, int n, int m, const Tensor points,
                                      Tensor temp, Tensor idx) {
  if (points.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(points);
    CHECK_CUDA_INPUT(temp);
    CHECK_CUDA_INPUT(idx);

    furthest_point_sampling_with_dist_cuda(b, n, m, points, temp, idx);
    return 1;
#else
    AT_ERROR(
        "furthest_point_sampling_with_dist is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("furthest_point_sampling_with_dist is not implemented on CPU");
  }
}
