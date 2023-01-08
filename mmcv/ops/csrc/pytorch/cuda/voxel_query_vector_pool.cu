// Copyright (c) OpenMMLab. All rights reserved.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pytorch_cuda_helper.hpp"
#include "vector_pool.cuh"
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

int StackVectorPoolForwardCUDAKernelLauncher(
    const Tensor support_xyz_tensor, const Tensor xyz_batch_cnt_tensor,
    const Tensor support_features_tensor, const Tensor new_xyz_tensor,
    const Tensor new_xyz_batch_cnt_tensor, Tensor new_features_tensor,
    Tensor new_local_xyz_tensor, Tensor point_cnt_of_grid_tensor,
    Tensor grouped_idxs_tensor, const int num_grid_x, const int num_grid_y,
    const int num_grid_z, const float max_neighbour_distance, const int use_xyz,
    const int num_max_sum_points, const int nsample, const int neighbor_type,
    const int pooling_type) {
  at::cuda::CUDAGuard device_guard(support_xyz_tensor.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const float *support_xyz = support_xyz_tensor.data<float>();
  const float *support_features = support_features_tensor.data<float>();
  const int *xyz_batch_cnt = xyz_batch_cnt_tensor.data<int>();
  const float *new_xyz = new_xyz_tensor.data<float>();
  const int *new_xyz_batch_cnt = new_xyz_batch_cnt_tensor.data<int>();
  float *new_features = new_features_tensor.data<float>();
  float *new_local_xyz = new_local_xyz_tensor.data<float>();
  int *point_cnt_of_grid = point_cnt_of_grid_tensor.data<int>();
  int *grouped_idxs = grouped_idxs_tensor.data<int>();
  int N = support_xyz_tensor.size(0);
  int batch_size = xyz_batch_cnt_tensor.size(0);
  int M = new_xyz_tensor.size(0);
  int num_c_out = new_features_tensor.size(1);
  int num_c_in = support_features_tensor.size(1);
  int num_total_grids = point_cnt_of_grid_tensor.size(1);

  int num_c_each_grid = num_c_out / num_total_grids;
  float grid_size_x = max_neighbour_distance * 2 / num_grid_x;
  float grid_size_y = max_neighbour_distance * 2 / num_grid_y;
  float grid_size_z = max_neighbour_distance * 2 / num_grid_z;
  dim3 blocks(DIVUP(M, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);
  int cum_sum = 0;
  int *p_cum_sum;
  cudaMalloc((void **)&p_cum_sum, sizeof(int));
  cudaMemcpy(p_cum_sum, &cum_sum, sizeof(int), cudaMemcpyHostToDevice);
  stack_vector_pool_cuda_kernel<<<blocks, threads>>>(
      support_xyz, support_features, xyz_batch_cnt, new_xyz, new_features,
      new_local_xyz, new_xyz_batch_cnt, num_grid_x, num_grid_y, num_grid_z,
      max_neighbour_distance, batch_size, M, num_c_in, num_c_out,
      num_c_each_grid, num_total_grids, point_cnt_of_grid, grouped_idxs,
      use_xyz, grid_size_x, grid_size_y, grid_size_z, p_cum_sum,
      num_max_sum_points, nsample, neighbor_type, pooling_type);

  cudaMemcpy(&cum_sum, p_cum_sum, sizeof(int), cudaMemcpyDeviceToHost);
  AT_CUDA_CHECK(cudaGetLastError());
  return cum_sum;
}

void StackVectorPoolBackwardCUDAKernelLauncher(
    const Tensor grad_new_features_tensor,
    const Tensor point_cnt_of_grid_tensor, const Tensor grouped_idxs_tensor,
    Tensor grad_support_features_tensor) {
  at::cuda::CUDAGuard device_guard(grad_new_features_tensor.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int M = grad_new_features_tensor.size(0);
  int num_c_out = grad_new_features_tensor.size(1);
  int N = grad_support_features_tensor.size(0);
  int num_c_in = grad_support_features_tensor.size(1);
  int num_total_grids = point_cnt_of_grid_tensor.size(1);
  int num_max_sum_points = grouped_idxs_tensor.size(0);

  const float *grad_new_features = grad_new_features_tensor.data<float>();
  const int *point_cnt_of_grid = point_cnt_of_grid_tensor.data<int>();
  const int *grouped_idxs = grouped_idxs_tensor.data<int>();
  float *grad_support_features = grad_support_features_tensor.data<float>();

  int num_c_each_grid = num_c_out / num_total_grids;
  dim3 blocks(DIVUP(num_max_sum_points, THREADS_PER_BLOCK),
              num_c_in);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);
  stack_vector_pool_backward_cuda_kernel<<<blocks, threads>>>(
      grad_new_features, point_cnt_of_grid, grouped_idxs, grad_support_features,
      N, M, num_c_out, num_c_in, num_c_each_grid, num_total_grids,
      num_max_sum_points);
  AT_CUDA_CHECK(cudaGetLastError());
}
