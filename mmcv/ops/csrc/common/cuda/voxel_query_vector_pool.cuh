#ifndef VOXEL_QUERY_VECTOR_POOL_CUH
#define VOXEL_QUERY_VECTOR_POOL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif


__global__ void stack_vector_pool_cuda_kernel(
    const float *support_xyz, const float *support_features,
    const int *xyz_batch_cnt, const float *new_xyz, float *new_features,
    float *new_local_xyz, const int *new_xyz_batch_cnt, int num_grid_x,
    int num_grid_y, int num_grid_z, float max_neighbour_distance,
    int batch_size, int M, int num_c_in, int num_c_out, int num_c_each_grid,
    int num_total_grids, int *point_cnt_of_grid, int *grouped_idxs, int use_xyz,
    float grid_size_x, float grid_size_y, float grid_size_z, int *cum_sum,
    int num_max_sum_points, int nsample, int neighbor_type, int pooling_type) {
  // support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
  // support_features: (N1 + N2 ..., C)
  // xyz_batch_cnt: (batch_size), [N1, N2, ...]
  // new_xyz: (M1 + M2 ..., 3) centers of the ball query
  // new_features: (M1 + M2 ..., C), C = num_total_grids * num_c_each_grid
  // new_local_xyz: (M1 + M2 ..., 3 * num_total_grids)
  // new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
  // num_grid_x, num_grid_y, num_grid_z: number of grids in each local area
  // centered at new_xyz point_cnt_of_grid: (M1 + M2 ..., num_total_grids)
  // grouped_idxs: (num_max_sum_points, 3)[idx of support_xyz, idx of new_xyz,
  // idx of grid_idx in new_xyz] use_xyz: whether to calculate new_local_xyz
  // neighbor_type: 1: ball, others: cube
  // pooling_type: 0: avg_pool, 1: random choice

  CUDA_1D_KERNEL_LOOP(pt_idx, M) {
    const float *cur_new_xyz = new_xyz;
    float *cur_new_features = new_features;
    int *cur_point_cnt_of_grid = point_cnt_of_grid;
    float *cur_new_local_xyz = new_local_xyz;
    const float *cur_support_xyz = support_xyz;
    const float *cur_support_features = support_features;
    int bs_idx = 0;
    for (int pt_cnt = 0; bs_idx < batch_size; bs_idx++) {
      pt_cnt += new_xyz_batch_cnt[bs_idx];
      if (pt_idx < pt_cnt) break;
    }

    int xyz_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) xyz_batch_start_idx += xyz_batch_cnt[k];

    cur_support_xyz += xyz_batch_start_idx * 3;
    cur_support_features += xyz_batch_start_idx * num_c_in;

    cur_new_xyz += pt_idx * 3;
    cur_new_features += pt_idx * num_c_out;
    cur_point_cnt_of_grid += pt_idx * num_total_grids;
    cur_new_local_xyz += pt_idx * 3 * num_total_grids;

    float new_x = cur_new_xyz[0];
    float new_y = cur_new_xyz[1];
    float new_z = cur_new_xyz[2];
    int n = xyz_batch_cnt[bs_idx], grid_idx_x, grid_idx_y, grid_idx_z, grid_idx;
    float local_x, local_y, local_z;
    float radius2 = max_neighbour_distance * max_neighbour_distance;

    int sample_cnt = 0;
    for (int k = 0; k < n; ++k) {
      local_x = cur_support_xyz[k * 3 + 0] - new_x;
      local_y = cur_support_xyz[k * 3 + 1] - new_y;
      local_z = cur_support_xyz[k * 3 + 2] - new_z;

      if (neighbor_type == 1) {
        // ball
        if (local_x * local_x + local_y * local_y + local_z * local_z >
            radius2) {
          continue;
        }
      } else {
        // voxel
        if ((fabs(local_x) > max_neighbour_distance) |
            (fabs(local_y) > max_neighbour_distance) |
            (fabs(local_z) > max_neighbour_distance)) {
          continue;
        }
      }

      grid_idx_x = floorf((local_x + max_neighbour_distance) / grid_size_x);
      grid_idx_y = floorf((local_y + max_neighbour_distance) / grid_size_y);
      grid_idx_z = floorf((local_z + max_neighbour_distance) / grid_size_z);
      grid_idx = grid_idx_x * num_grid_y * num_grid_z +
                 grid_idx_y * num_grid_z + grid_idx_z;
      grid_idx = min(max(grid_idx, 0), num_total_grids - 1);

      if (pooling_type == 0) {
        // avg pooling
        cur_point_cnt_of_grid[grid_idx]++;

        for (int i = 0; i < num_c_in; i++) {
          cur_new_features[grid_idx * num_c_each_grid + i % num_c_each_grid] +=
              cur_support_features[k * num_c_in + i];
        }
        if (use_xyz) {
          cur_new_local_xyz[grid_idx * 3 + 0] += local_x;
          cur_new_local_xyz[grid_idx * 3 + 1] += local_y;
          cur_new_local_xyz[grid_idx * 3 + 2] += local_z;
        }

        int cnt = atomicAdd(cum_sum, 1);
        if (cnt >= num_max_sum_points)
          continue;  // continue to statistics the max number of points

        grouped_idxs[cnt * 3 + 0] = xyz_batch_start_idx + k;
        grouped_idxs[cnt * 3 + 1] = pt_idx;
        grouped_idxs[cnt * 3 + 2] = grid_idx;

        sample_cnt++;
        if (nsample > 0 && sample_cnt >= nsample) break;
      } else if (pooling_type == 1) {
        // random choose one within sub-voxel
        // printf("new_xyz=(%.2f, %.2f, %.2f, ), find neighbor k=%d:
        // support_xyz=(%.2f, %.2f, %.2f), local_xyz=(%.2f, %.2f, %.2f),
        // neighbor=%.2f, grid_idx=%d, point_cnt_of_grid_idx=%d\n", new_x,
        // new_y, new_z, k, support_xyz[k * 3 + 0], support_xyz[k * 3 + 1],
        // support_xyz[k * 3 + 2], local_x, local_y, local_z,
        // max_neighbour_distance, grid_idx, point_cnt_of_grid[grid_idx]);

        if (point_cnt_of_grid[grid_idx] == 0) {
          point_cnt_of_grid[grid_idx]++;
          for (int i = 0; i < num_c_in; i++) {
            cur_new_features[grid_idx * num_c_each_grid + i % num_c_each_grid] =
                cur_support_features[k * num_c_in + i];
          }
          if (use_xyz) {
            cur_new_local_xyz[grid_idx * 3 + 0] = local_x;
            cur_new_local_xyz[grid_idx * 3 + 1] = local_y;
            cur_new_local_xyz[grid_idx * 3 + 2] = local_z;
          }

          int cnt = atomicAdd(cum_sum, 1);
          if (cnt >= num_max_sum_points)
            continue;  // continue to statistics the max number of points

          grouped_idxs[cnt * 3 + 0] = xyz_batch_start_idx + k;
          grouped_idxs[cnt * 3 + 1] = pt_idx;
          grouped_idxs[cnt * 3 + 2] = grid_idx;

          sample_cnt++;
          if (nsample > 0 && sample_cnt >= nsample ||
              sample_cnt >= num_total_grids)
            break;
        }
      }
    }
  }
}

__global__ void stack_vector_pool_backward_cuda_kernel(
    const float *grad_new_features, const int *point_cnt_of_grid,
    const int *grouped_idxs, float *grad_support_features, int N, int M,
    int num_c_out, int num_c_in, int num_c_each_grid, int num_total_grids,
    int num_max_sum_points) {
  // grad_new_features: (M1 + M2 ..., C_out)
  // point_cnt_of_grid: (M1 + M2 ..., num_total_grids)
  // grouped_idxs: (num_max_sum_points, 3) [idx of support_xyz, idx of new_xyz,
  // idx of grid_idx in new_xyz] grad_support_features: (N1 + N2 ..., C_in)

  int channel_idx = blockIdx.y;
  if (channel_idx >= num_c_in) return;
  CUDA_1D_KERNEL_LOOP(index, num_max_sum_points) {
    const float *cur_grad_new_features = grad_new_features;
    float *cur_grad_support_features = grad_support_features;

    int idx_of_support_xyz = grouped_idxs[index * 3 + 0];
    int idx_of_new_xyz = grouped_idxs[index * 3 + 1];
    int idx_of_grid_idx = grouped_idxs[index * 3 + 2];

    int num_total_pts =
        point_cnt_of_grid[idx_of_new_xyz * num_total_grids + idx_of_grid_idx];
    cur_grad_support_features += idx_of_support_xyz * num_c_in + channel_idx;

    cur_grad_new_features +=
        idx_of_new_xyz * num_c_out + idx_of_grid_idx * num_c_each_grid;
    int channel_idx_of_cin = channel_idx % num_c_each_grid;
    float cur_grad = 1 / fmaxf(float(num_total_pts), 1.0);
    atomicAdd(cur_grad_support_features,
              cur_grad_new_features[channel_idx_of_cin] * cur_grad);
  }
}

#endif
