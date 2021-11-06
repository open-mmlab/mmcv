// Copyright (c) OpenMMLab. All rights reserved.
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

int hard_voxelize_forward_impl(const at::Tensor& points, at::Tensor& voxels,
                               at::Tensor& coors,
                               at::Tensor& num_points_per_voxel,
                               const std::vector<float> voxel_size,
                               const std::vector<float> coors_range,
                               const int max_points, const int max_voxels,
                               const int NDim = 3) {
  return DISPATCH_DEVICE_IMPL(hard_voxelize_forward_impl, points, voxels, coors,
                              num_points_per_voxel, voxel_size, coors_range,
                              max_points, max_voxels, NDim);
}

void dynamic_voxelize_forward_impl(const at::Tensor& points, at::Tensor& coors,
                                   const std::vector<float> voxel_size,
                                   const std::vector<float> coors_range,
                                   const int NDim = 3) {
  DISPATCH_DEVICE_IMPL(dynamic_voxelize_forward_impl, points, coors, voxel_size,
                       coors_range, NDim);
}

int hard_voxelize_forward(const at::Tensor& points, at::Tensor& voxels,
                          at::Tensor& coors, at::Tensor& num_points_per_voxel,
                          const std::vector<float> voxel_size,
                          const std::vector<float> coors_range,
                          const int max_points, const int max_voxels,
                          const int NDim = 3) {
  return hard_voxelize_forward_impl(points, voxels, coors, num_points_per_voxel,
                                    voxel_size, coors_range, max_points,
                                    max_voxels, NDim);
}

void dynamic_voxelize_forward(const at::Tensor& points, at::Tensor& coors,
                              const std::vector<float> voxel_size,
                              const std::vector<float> coors_range,
                              const int NDim = 3) {
  dynamic_voxelize_forward_impl(points, coors, voxel_size, coors_range, NDim);
}
