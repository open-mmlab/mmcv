// Copyright (c) OpenMMLab. All rights reserved.
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include "csrc_dipu/diopirt/diopirt_impl.h"
#include "csrc_dipu/base/basedef.h"

using dipu::diopi_helper::toDiopiScalar;
using dipu::diopi_helper::toDiopiTensorHandle;
#endif

int hard_voxelize_forward_impl(const at::Tensor &points, at::Tensor &voxels,
                               at::Tensor &coors,
                               at::Tensor &num_points_per_voxel,
                               const std::vector<float> voxel_size,
                               const std::vector<float> coors_range,
                               const int max_points, const int max_voxels,
                               const int NDim = 3) {
  return DISPATCH_DEVICE_IMPL(hard_voxelize_forward_impl, points, voxels, coors,
                              num_points_per_voxel, voxel_size, coors_range,
                              max_points, max_voxels, NDim);
}

int nondeterministic_hard_voxelize_forward_impl(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim = 3) {
  return DISPATCH_DEVICE_IMPL(nondeterministic_hard_voxelize_forward_impl,
                              points, voxels, coors, num_points_per_voxel,
                              voxel_size, coors_range, max_points, max_voxels,
                              NDim);
}

void dynamic_voxelize_forward_impl(const at::Tensor &points, at::Tensor &coors,
                                   const std::vector<float> voxel_size,
                                   const std::vector<float> coors_range,
                                   const int NDim = 3) {
  DISPATCH_DEVICE_IMPL(dynamic_voxelize_forward_impl, points, coors, voxel_size,
                       coors_range, NDim);
}

#ifdef MMCV_WITH_DIOPI
void hard_voxelize_forward_diopi(const at::Tensor &points,
                                 const at::Tensor &voxel_size,
                                 const at::Tensor &coors_range,
                                 at::Tensor &voxels, at::Tensor &coors,
                                 at::Tensor &num_points_per_voxel,
                                 at::Tensor &voxel_num, const int max_points,
                                 const int max_voxels, const int NDim = 3,
                                 const bool deterministic = true) {
  auto points_p = toDiopiTensorHandle(points);
  diopiDevice_t device;
  diopiGetTensorDevice(points_p, &device);
  if (device == diopi_host) {
    int64_t *voxel_num_data = voxel_num.data_ptr<int64_t>();
    std::vector<float> voxel_size_v(
        voxel_size.data_ptr<float>(),
        voxel_size.data_ptr<float>() + voxel_size.numel());
    std::vector<float> coors_range_v(
        coors_range.data_ptr<float>(),
        coors_range.data_ptr<float>() + coors_range.numel());

    if (deterministic) {
      *voxel_num_data = hard_voxelize_forward_impl(
          points, voxels, coors, num_points_per_voxel, voxel_size_v,
          coors_range_v, max_points, max_voxels, NDim);
    } else {
      TORCH_CHECK(
          deterministic,
          "nondeterministic hard_voxelize_forward is not supported on host!");
    }
    return;
  }
  diopiContext ctx(dipu::getCurrentDIPUStream().rawstream());
  diopiContextHandle_t ch = &ctx;
  auto voxel_size_p = toDiopiTensorHandle(voxel_size);
  auto coors_range_p = toDiopiTensorHandle(coors_range);
  auto voxels_p = toDiopiTensorHandle(voxels);
  auto coors_p = toDiopiTensorHandle(coors);
  auto num_points_per_voxel_p = toDiopiTensorHandle(num_points_per_voxel);
  auto voxel_num_p = toDiopiTensorHandle(voxel_num);
  bool is_mock_cuda = points.device().type() == dipu::DIPU_DEVICE_TYPE;
  if (is_mock_cuda && reinterpret_cast<void *>(diopiHardVoxelizeMmcv) != nullptr) {
    auto ret = diopiHardVoxelizeMmcv(
        ch, voxels_p, coors_p, num_points_per_voxel_p, voxel_num_p, points_p,
        voxel_size_p, coors_range_p, max_points, max_voxels, NDim,
        deterministic);
    if (ret == diopiSuccess) return;
  }
  LOG(WARNING) << "Fallback to cpu: mmcv ext op hard_voxelize_forward";
  auto points_cpu = points.cpu();
  auto voxel_size_cpu = voxel_size.cpu();
  auto coors_range_cpu = coors_range.cpu();
  auto voxels_cpu = voxels.cpu();
  auto coors_cpu = coors.cpu();
  auto num_points_per_voxel_cpu = num_points_per_voxel.cpu();
  auto voxel_num_cpu = voxel_num.cpu();

  int64_t *voxel_num_data_cpu = voxel_num_cpu.data_ptr<int64_t>();
  std::vector<float> voxel_size_v_cpu(
      voxel_size_cpu.data_ptr<float>(),
      voxel_size_cpu.data_ptr<float>() + voxel_size_cpu.numel());
  std::vector<float> coors_range_v_cpu(
      coors_range_cpu.data_ptr<float>(),
      coors_range_cpu.data_ptr<float>() + coors_range_cpu.numel());

  if (deterministic) {
    *voxel_num_data_cpu = hard_voxelize_forward_impl(
        points_cpu, voxels_cpu, coors_cpu, num_points_per_voxel_cpu,
        voxel_size_v_cpu, coors_range_v_cpu, max_points, max_voxels, NDim);
  } else {
    puts("nondeterministic hard_voxelize_forward is not supported on host!");
    abort();
  }
  voxels.copy_(voxels_cpu);
  coors.copy_(coors_cpu);
  num_points_per_voxel.copy_(num_points_per_voxel_cpu);
  voxel_num.copy_(voxel_num_cpu);
  return;
}

void dynamic_voxelize_forward_diopi(const at::Tensor &points,
                                    const at::Tensor &voxel_size,
                                    const at::Tensor &coors_range,
                                    at::Tensor &coors, const int NDim = 3) {
  auto points_p = toDiopiTensorHandle(points);
  diopiDevice_t device;
  diopiGetTensorDevice(points_p, &device);
  if (device == diopi_host) {
    std::vector<float> voxel_size_v(
        voxel_size.data_ptr<float>(),
        voxel_size.data_ptr<float>() + voxel_size.numel());
    std::vector<float> coors_range_v(
        coors_range.data_ptr<float>(),
        coors_range.data_ptr<float>() + coors_range.numel());
    dynamic_voxelize_forward_impl(points, coors, voxel_size_v, coors_range_v,
                                  NDim);
    return;
  }
  diopiContext ctx(dipu::getCurrentDIPUStream().rawstream());
  diopiContextHandle_t ch = &ctx;
  auto voxel_size_p = toDiopiTensorHandle(voxel_size);
  auto coors_range_p = toDiopiTensorHandle(coors_range);
  auto coors_p = toDiopiTensorHandle(coors);
  bool is_mock_cuda = points.device().type() == dipu::DIPU_DEVICE_TYPE;
  if (is_mock_cuda && reinterpret_cast<void *>(diopiDynamicVoxelizeMmcv) != nullptr) {
    auto ret = diopiDynamicVoxelizeMmcv(ch, coors_p, points_p, voxel_size_p,
                                        coors_range_p, NDim);
    if (ret == diopiSuccess) return;
  }
  LOG(WARNING) << "Fallback to cpu: mmcv ext op dynamic_voxelize_forward";
  auto points_cpu = points.cpu();
  auto voxel_size_cpu = voxel_size.cpu();
  auto coors_range_cpu = coors_range.cpu();
  auto coors_cpu = coors.cpu();

  std::vector<float> voxel_size_v_cpu(
      voxel_size_cpu.data_ptr<float>(),
      voxel_size_cpu.data_ptr<float>() + voxel_size_cpu.numel());
  std::vector<float> coors_range_v_cpu(
      coors_range_cpu.data_ptr<float>(),
      coors_range_cpu.data_ptr<float>() + coors_range_cpu.numel());
  dynamic_voxelize_forward_impl(points_cpu, coors_cpu, voxel_size_v_cpu,
                                coors_range_v_cpu, NDim);
  coors.copy_(coors_cpu);
  return;
}
#endif

void hard_voxelize_forward(const at::Tensor &points,
                           const at::Tensor &voxel_size,
                           const at::Tensor &coors_range, at::Tensor &voxels,
                           at::Tensor &coors, at::Tensor &num_points_per_voxel,
                           at::Tensor &voxel_num, const int max_points,
                           const int max_voxels, const int NDim = 3,
                           const bool deterministic = true) {
#ifdef MMCV_WITH_DIOPI
  hard_voxelize_forward_diopi(points, voxel_size, coors_range, voxels, coors,
                              num_points_per_voxel, voxel_num, max_points,
                              max_voxels, NDim, deterministic);
#else
  int64_t *voxel_num_data = voxel_num.data_ptr<int64_t>();
  std::vector<float> voxel_size_v(
      voxel_size.data_ptr<float>(),
      voxel_size.data_ptr<float>() + voxel_size.numel());
  std::vector<float> coors_range_v(
      coors_range.data_ptr<float>(),
      coors_range.data_ptr<float>() + coors_range.numel());

  if (deterministic) {
    *voxel_num_data = hard_voxelize_forward_impl(
        points, voxels, coors, num_points_per_voxel, voxel_size_v,
        coors_range_v, max_points, max_voxels, NDim);
  } else {
    *voxel_num_data = nondeterministic_hard_voxelize_forward_impl(
        points, voxels, coors, num_points_per_voxel, voxel_size_v,
        coors_range_v, max_points, max_voxels, NDim);
  }
#endif
}

void dynamic_voxelize_forward(const at::Tensor &points,
                              const at::Tensor &voxel_size,
                              const at::Tensor &coors_range, at::Tensor &coors,
                              const int NDim = 3) {
#ifdef MMCV_WITH_DIOPI
  dynamic_voxelize_forward_diopi(points, voxel_size, coors_range, coors, NDim);
#else
  std::vector<float> voxel_size_v(
      voxel_size.data_ptr<float>(),
      voxel_size.data_ptr<float>() + voxel_size.numel());
  std::vector<float> coors_range_v(
      coors_range.data_ptr<float>(),
      coors_range.data_ptr<float>() + coors_range.numel());
  dynamic_voxelize_forward_impl(points, coors, voxel_size_v, coors_range_v,
                                NDim);
#endif
}
