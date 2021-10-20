// Copyright (c) OpenMMLab. All rights reserved.
#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
int HardVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim = 3);

int hard_voxelize_forward_cuda(const at::Tensor &points, at::Tensor &voxels,
                               at::Tensor &coors,
                               at::Tensor &num_points_per_voxel,
                               const std::vector<float> voxel_size,
                               const std::vector<float> coors_range,
                               const int max_points, const int max_voxels,
                               const int NDim = 3) {
  return HardVoxelizeForwardCUDAKernelLauncher(
      points, voxels, coors, num_points_per_voxel, voxel_size, coors_range,
      max_points, max_voxels, NDim);
};

void DynamicVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor &points, at::Tensor &coors,
    const std::vector<float> voxel_size, const std::vector<float> coors_range,
    const int NDim = 3);

void dynamic_voxelize_forward_cuda(const at::Tensor &points, at::Tensor &coors,
                                   const std::vector<float> voxel_size,
                                   const std::vector<float> coors_range,
                                   const int NDim = 3) {
  DynamicVoxelizeForwardCUDAKernelLauncher(points, coors, voxel_size,
                                           coors_range, NDim);
};
#endif

int hard_voxelize_forward_cpu(const at::Tensor &points, at::Tensor &voxels,
                              at::Tensor &coors,
                              at::Tensor &num_points_per_voxel,
                              const std::vector<float> voxel_size,
                              const std::vector<float> coors_range,
                              const int max_points, const int max_voxels,
                              const int NDim = 3);

void dynamic_voxelize_forward_cpu(const at::Tensor &points, at::Tensor &coors,
                                  const std::vector<float> voxel_size,
                                  const std::vector<float> coors_range,
                                  const int NDim = 3);

int hard_voxelize_forward(const at::Tensor &points, at::Tensor &voxels,
                          at::Tensor &coors, at::Tensor &num_points_per_voxel,
                          const std::vector<float> voxel_size,
                          const std::vector<float> coors_range,
                          const int max_points, const int max_voxels,
                          const int NDim = 3) {
  if (points.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(points);

    return hard_voxelize_forward_cuda(
        points, voxels, coors, num_points_per_voxel, voxel_size, coors_range,
        max_points, max_voxels, NDim);
#else
    AT_ERROR("hard_voxelize is not compiled with GPU support");
#endif
  } else {
    return hard_voxelize_forward_cpu(points, voxels, coors,
                                     num_points_per_voxel, voxel_size,
                                     coors_range, max_points, max_voxels, NDim);
  }
}

void dynamic_voxelize_forward(const at::Tensor &points, at::Tensor &coors,
                              const std::vector<float> voxel_size,
                              const std::vector<float> coors_range,
                              const int NDim = 3) {
  if (points.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(points);

    dynamic_voxelize_forward_cuda(points, coors, voxel_size, coors_range, NDim);
#else
    AT_ERROR("dynamic_voxelize is not compiled with GPU support");
#endif
  } else {
    dynamic_voxelize_forward_cpu(points, coors, voxel_size, coors_range, NDim);
  }
}
