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

void hard_voxelize_forward(const at::Tensor &points,
                           const at::Tensor &voxel_size,
                           const at::Tensor &coors_range, at::Tensor &voxels,
                           at::Tensor &coors, at::Tensor &num_points_per_voxel,
                           at::Tensor &voxel_num, const int max_points,
                           const int max_voxels, const int NDim = 3) {
  int64_t *voxel_num_data = voxel_num.data_ptr<int64_t>();
  std::vector<float> voxel_size_v(
      voxel_size.data_ptr<float>(),
      voxel_size.data_ptr<float>() + voxel_size.numel());
  std::vector<float> coors_range_v(
      coors_range.data_ptr<float>(),
      coors_range.data_ptr<float>() + coors_range.numel());
  if (points.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(points);

    *voxel_num_data = hard_voxelize_forward_cuda(
        points, voxels, coors, num_points_per_voxel, voxel_size_v,
        coors_range_v, max_points, max_voxels, NDim);
#else
    AT_ERROR("hard_voxelize is not compiled with GPU support");
#endif
  } else {
    *voxel_num_data = hard_voxelize_forward_cpu(
        points, voxels, coors, num_points_per_voxel, voxel_size_v,
        coors_range_v, max_points, max_voxels, NDim);
  }
}

void dynamic_voxelize_forward(const at::Tensor &points,
                              const at::Tensor &voxel_size,
                              const at::Tensor &coors_range, at::Tensor &coors,
                              const int NDim = 3) {
  std::vector<float> voxel_size_v(
      voxel_size.data_ptr<float>(),
      voxel_size.data_ptr<float>() + voxel_size.numel());
  std::vector<float> coors_range_v(
      coors_range.data_ptr<float>(),
      coors_range.data_ptr<float>() + coors_range.numel());
  if (points.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(points);

    dynamic_voxelize_forward_cuda(points, coors, voxel_size_v, coors_range_v,
                                  NDim);
#else
    AT_ERROR("dynamic_voxelize is not compiled with GPU support");
#endif
  } else {
    dynamic_voxelize_forward_cpu(points, coors, voxel_size_v, coors_range_v,
                                 NDim);
  }
}
