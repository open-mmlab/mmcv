// Copyright (c) OpenMMLab. All rights reserved.
#include <stdio.h>
#include <stdlib.h>

#include "pytorch_cuda_helper.hpp"
#include "voxelization_cuda_kernel.cuh"


namespace {

static int HardVoxelizeForwardV1CUDAKernelLauncher(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim = 3) {
  // current version tooks about 0.04s for one frame on cpu
  // check device

  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int num_points = points.size(0);
  const int num_features = points.size(1);

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  // map points to voxel coors
  at::Tensor temp_coors =
      at::zeros({num_points, NDim}, points.options().dtype(at::kInt));

  dim3 grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
  dim3 block(512);

  // 1. link point to corresponding voxel coors
  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "hard_voxelize_kernel", ([&] {
        dynamic_voxelize_kernel<scalar_t, int><<<grid, block, 0, stream>>>(
            points.contiguous().data_ptr<scalar_t>(),
            temp_coors.contiguous().data_ptr<int>(), voxel_x, voxel_y, voxel_z,
            coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max,
            coors_z_max, grid_x, grid_y, grid_z, num_points, num_features,
            NDim);
      }));

  AT_CUDA_CHECK(cudaGetLastError());

  // 2. map point to the idx of the corresponding voxel, find duplicate coor
  // create some temporary variables
  auto point_to_pointidx = -at::ones(
      {
          num_points,
      },
      points.options().dtype(at::kInt));
  auto point_to_voxelidx = -at::ones(
      {
          num_points,
      },
      points.options().dtype(at::kInt));

  dim3 map_grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
  dim3 map_block(512);

  AT_DISPATCH_ALL_TYPES(
      temp_coors.scalar_type(), "determin_duplicate", ([&] {
        point_to_voxelidx_kernel<int><<<map_grid, map_block, 0, stream>>>(
            temp_coors.contiguous().data_ptr<int>(),
            point_to_voxelidx.contiguous().data_ptr<int>(),
            point_to_pointidx.contiguous().data_ptr<int>(), max_points,
            max_voxels, num_points, NDim);
      }));

  AT_CUDA_CHECK(cudaGetLastError());

  // 3. determine voxel num and voxel's coor index
  // make the logic in the CUDA device could accelerate about 10 times
  auto coor_to_voxelidx = -at::ones(
      {
          num_points,
      },
      points.options().dtype(at::kInt));
  auto voxel_num = at::zeros(
      {
          1,
      },
      points.options().dtype(at::kInt));  // must be zero from the beginning

  AT_DISPATCH_ALL_TYPES(temp_coors.scalar_type(), "determin_duplicate", ([&] {
                          determin_voxel_num<int><<<1, 1, 0, stream>>>(
                              num_points_per_voxel.contiguous().data_ptr<int>(),
                              point_to_voxelidx.contiguous().data_ptr<int>(),
                              point_to_pointidx.contiguous().data_ptr<int>(),
                              coor_to_voxelidx.contiguous().data_ptr<int>(),
                              voxel_num.contiguous().data_ptr<int>(),
                              max_points, max_voxels, num_points);
                        }));

  AT_CUDA_CHECK(cudaGetLastError());

  // 4. copy point features to voxels
  // Step 4 & 5 could be parallel
  auto pts_output_size = num_points * num_features;
  dim3 cp_grid(std::min(at::cuda::ATenCeilDiv(pts_output_size, 512), 4096));
  dim3 cp_block(512);
  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "assign_point_to_voxel", ([&] {
        assign_point_to_voxel<float, int><<<cp_grid, cp_block, 0, stream>>>(
            pts_output_size, points.contiguous().data_ptr<float>(),
            point_to_voxelidx.contiguous().data_ptr<int>(),
            coor_to_voxelidx.contiguous().data_ptr<int>(),
            voxels.contiguous().data_ptr<float>(), max_points, num_features,
            num_points, NDim);
      }));
  //   cudaDeviceSynchronize();
  //   AT_CUDA_CHECK(cudaGetLastError());

  // 5. copy coors of each voxels
  auto coors_output_size = num_points * NDim;
  dim3 coors_cp_grid(
      std::min(at::cuda::ATenCeilDiv(coors_output_size, 512), 4096));
  dim3 coors_cp_block(512);
  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "assign_point_to_voxel", ([&] {
        assign_voxel_coors<float, int>
            <<<coors_cp_grid, coors_cp_block, 0, stream>>>(
                coors_output_size, temp_coors.contiguous().data_ptr<int>(),
                point_to_voxelidx.contiguous().data_ptr<int>(),
                coor_to_voxelidx.contiguous().data_ptr<int>(),
                coors.contiguous().data_ptr<int>(), num_points, NDim);
      }));

  AT_CUDA_CHECK(cudaGetLastError());

  auto voxel_num_cpu = voxel_num.to(at::kCPU);
  int voxel_num_int = voxel_num_cpu.data_ptr<int>()[0];

  return voxel_num_int;
}

}  // namespace

// Threads per block and block cap for the optimized kernels.
constexpr int kV2Threads = 256;
constexpr int kV2MaxBlocks = 4096;

int HardVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim = 3) {
  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int num_points = points.size(0);
  const int num_features = points.size(1);
  if (num_points == 0) return 0;

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  // The optimized path is a 3-D perfect hash over (z, y, x) and works on
  // float32 points only.  Everything else (other NDim, other dtypes) falls
  // back to the legacy implementation with identical semantics.
  if (NDim != 3 || points.scalar_type() != at::kFloat) {
    return HardVoxelizeForwardV1CUDAKernelLauncher(
        points, voxels, coors, num_points_per_voxel, voxel_size, coors_range,
        max_points, max_voxels, NDim);
  }

  const int64_t grid_total = (int64_t)grid_x * grid_y * grid_z;
  TORCH_CHECK(grid_total <= INT32_MAX,
              "hard_voxelize: grid dims overflow int32 lin index.");

  const auto opts = points.options();
  at::Tensor points_contig = points.contiguous();

  // K1: per-point linear voxel id (out-of-range -> -1).  Float32 arithmetic
  // is identical to dynamic_voxelize_kernel, so valid lin values match the
  // legacy coors exactly.
  at::Tensor lin = at::empty({num_points}, opts.dtype(at::kInt));
  {
    const int blocks =
        std::min(at::cuda::ATenCeilDiv(num_points, kV2Threads), kV2MaxBlocks);
    voxel_index_kernel<int><<<blocks, kV2Threads, 0, stream>>>(
        points_contig.data_ptr<float>(), num_points, num_features, voxel_x,
        voxel_y, voxel_z, coors_x_min, coors_y_min, coors_z_min, grid_x,
        grid_y, grid_z, lin.data_ptr<int>());
    AT_CUDA_CHECK(cudaGetLastError());
  }

  // K2a: stable sort by lin.  Equal lin keeps the original relative order,
  // so the first point of each group is the one that appeared first in the
  // input, which is what the legacy determin_voxel_num numbering depends on.
  at::Tensor order =
      at::argsort(lin, /*stable=*/true, /*dim=*/0, /*descending=*/false);
  at::Tensor lin_sorted_all = lin.index({order});
  const int64_t num_invalid = (lin_sorted_all < 0).sum().item<int64_t>();
  const int64_t num_valid = num_points - num_invalid;
  if (num_valid == 0) return 0;

  at::Tensor lin_sorted =
      lin_sorted_all.slice(0, num_invalid, num_points).contiguous();
  at::Tensor orig_idx = order.slice(0, num_invalid, num_points).contiguous();

  // K2b: group boundaries (is_start) -> group id via cumsum.
  at::Tensor is_start = at::ones({num_valid}, opts.dtype(at::kBool));
  if (num_valid > 1) {
    is_start.slice(0, 1, num_valid)
        .copy_(lin_sorted.slice(0, 1, num_valid) !=
               lin_sorted.slice(0, 0, num_valid - 1));
  }
  at::Tensor group_id = at::cumsum(is_start.to(at::kLong), 0).sub(1);
  const int64_t num_groups = is_start.sum().item<int64_t>();
  TORCH_CHECK(num_groups > 0, "hard_voxelize: no valid voxel group.");

  // K2c: group_start[G + 1] with a sentinel at [G] = num_valid, and the
  // original-row index of each group head.
  at::Tensor group_start = at::empty({num_groups + 1}, opts.dtype(at::kInt));
  at::Tensor leader_orig = at::empty({num_groups}, opts.dtype(at::kLong));
  {
    const int blocks =
        at::cuda::ATenCeilDiv(static_cast<int>(num_valid) + 1, kV2Threads);
    group_meta_lead_kernel<int><<<blocks, kV2Threads, 0, stream>>>(
        group_id.data_ptr<int64_t>(), is_start.data_ptr<bool>(),
        orig_idx.data_ptr<int64_t>(), (int)num_valid, (int)num_groups,
        group_start.data_ptr<int>(), leader_orig.data_ptr<int64_t>());
    AT_CUDA_CHECK(cudaGetLastError());
  }

  // K2d: number voxels by first-occurrence order (ascending leader row id),
  // identical to the legacy single-threaded numbering.
  at::Tensor vorder = at::argsort(leader_orig, /*stable=*/true, /*dim=*/0,
                                  /*descending=*/false);
  at::Tensor voxel_id = at::empty({num_groups}, opts.dtype(at::kLong));
  voxel_id.index_copy_(0, vorder,
                       at::arange(num_groups, opts.dtype(at::kLong)));

  // K3: scatter into the pre-allocated (max_voxels-sized) buffers.
  const int64_t voxel_num = std::min<int64_t>(num_groups, max_voxels);
  if (voxel_num > 0) {
    const int blocks = std::min(
        at::cuda::ATenCeilDiv(static_cast<int>(num_valid), kV2Threads),
        kV2MaxBlocks);
    scatter_kernel<int><<<blocks, kV2Threads, 0, stream>>>(
        points_contig.data_ptr<float>(), num_features,
        orig_idx.data_ptr<int64_t>(), group_id.data_ptr<int64_t>(),
        lin_sorted.data_ptr<int>(), group_start.data_ptr<int>(),
        voxel_id.data_ptr<int64_t>(), (int)num_valid, max_points, max_voxels,
        grid_x, grid_y, voxels.data_ptr<float>(), coors.data_ptr<int>(),
        num_points_per_voxel.data_ptr<int>());
    AT_CUDA_CHECK(cudaGetLastError());
  }
  return (int)voxel_num;
}

int NondeterministicHardVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim = 3) {
  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int num_points = points.size(0);
  const int num_features = points.size(1);

  if (num_points == 0) return 0;

  dim3 blocks(
      std::min(at::cuda::ATenCeilDiv(num_points, THREADS_PER_BLOCK), 4096));
  dim3 threads(THREADS_PER_BLOCK);

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  // map points to voxel coors
  at::Tensor temp_coors =
      at::zeros({num_points, NDim}, points.options().dtype(at::kInt));

  // 1. link point to corresponding voxel coors
  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "hard_voxelize_kernel", ([&] {
        dynamic_voxelize_kernel<scalar_t, int><<<blocks, threads, 0, stream>>>(
            points.contiguous().data_ptr<scalar_t>(),
            temp_coors.contiguous().data_ptr<int>(), voxel_x, voxel_y, voxel_z,
            coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max,
            coors_z_max, grid_x, grid_y, grid_z, num_points, num_features,
            NDim);
      }));

  at::Tensor coors_map;
  at::Tensor reduce_count;

  auto coors_clean = temp_coors.masked_fill(temp_coors.lt(0).any(-1, true), -1);

  std::tie(temp_coors, coors_map, reduce_count) =
      at::unique_dim(coors_clean, 0, true, true, false);

  if (temp_coors[0][0].lt(0).item<bool>()) {
    // the first element of temp_coors is (-1,-1,-1) and should be removed
    temp_coors = temp_coors.slice(0, 1);
    coors_map = coors_map - 1;
  }

  int num_coors = temp_coors.size(0);
  temp_coors = temp_coors.to(at::kInt);
  coors_map = coors_map.to(at::kInt);

  at::Tensor coors_count = at::zeros({1}, coors_map.options());
  at::Tensor coors_order = at::empty({num_coors}, coors_map.options());
  at::Tensor pts_id = at::zeros({num_points}, coors_map.options());
  reduce_count = at::zeros({num_coors}, coors_map.options());

  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "get_assign_pos", ([&] {
        nondeterministic_get_assign_pos<<<blocks, threads, 0, stream>>>(
            num_points, coors_map.contiguous().data_ptr<int32_t>(),
            pts_id.contiguous().data_ptr<int32_t>(),
            coors_count.contiguous().data_ptr<int32_t>(),
            reduce_count.contiguous().data_ptr<int32_t>(),
            coors_order.contiguous().data_ptr<int32_t>());
      }));

  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "assign_point_to_voxel", ([&] {
        nondeterministic_assign_point_voxel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                num_points, points.contiguous().data_ptr<scalar_t>(),
                coors_map.contiguous().data_ptr<int32_t>(),
                pts_id.contiguous().data_ptr<int32_t>(),
                temp_coors.contiguous().data_ptr<int32_t>(),
                reduce_count.contiguous().data_ptr<int32_t>(),
                coors_order.contiguous().data_ptr<int32_t>(),
                voxels.contiguous().data_ptr<scalar_t>(),
                coors.contiguous().data_ptr<int32_t>(),
                num_points_per_voxel.contiguous().data_ptr<int32_t>(),
                max_voxels, max_points, num_features, NDim);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
  return max_voxels < num_coors ? max_voxels : num_coors;
}

void DynamicVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor &points, at::Tensor &coors,
    const std::vector<float> voxel_size, const std::vector<float> coors_range,
    const int NDim = 3) {
  // current version tooks about 0.04s for one frame on cpu
  // check device

  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int num_points = points.size(0);
  const int num_features = points.size(1);

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  const int col_blocks = at::cuda::ATenCeilDiv(num_points, THREADS_PER_BLOCK);
  dim3 blocks(col_blocks);
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_ALL_TYPES(points.scalar_type(), "dynamic_voxelize_kernel", [&] {
    dynamic_voxelize_kernel<scalar_t, int><<<blocks, threads, 0, stream>>>(
        points.contiguous().data_ptr<scalar_t>(),
        coors.contiguous().data_ptr<int>(), voxel_x, voxel_y, voxel_z,
        coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max,
        coors_z_max, grid_x, grid_y, grid_z, num_points, num_features, NDim);
  });

  AT_CUDA_CHECK(cudaGetLastError());
}
