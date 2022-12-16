// Copyright (c) OpenMMLab. All rights reserved.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pytorch_cuda_helper.hpp"
#include "vector_pool.cuh"
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))


void StackQueryLocalNeighborIdxsCUDAKernelLauncher(
    const Tensor support_xyz_tensor, const Tensor xyz_batch_cnt_tensor,
    const Tensor new_xyz_tensor, const Tensor new_xyz_batch_cnt_tensor,
    Tensor stack_neighbor_idxs_tensor, Tensor start_len_tensor,
    Tensor cumsum_tensor, const int avg_length_of_neighbor_idxs,
    const float max_neighbour_distance, const int nsample,
    const int neighbor_type) {
  int batch_size = xyz_batch_cnt_tensor.size(0);
  int M = new_xyz_tensor.size(0);
  at::cuda::CUDAGuard device_guard(support_xyz_tensor.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(DIVUP(M, THREADS_PER_BLOCK));
  dim3 threads(THREADS_PER_BLOCK);

  query_stacked_local_neighbor_idxs_cuda_kernel<<<blocks, threads>>>(
      support_xyz_tensor.data_ptr<float>(),
      xyz_batch_cnt_tensor.data_ptr<int>(), new_xyz_tensor.data_ptr<float>(),
      new_xyz_batch_cnt_tensor.data_ptr<int>(),
      stack_neighbor_idxs_tensor.data_ptr<int>(),
      start_len_tensor.data_ptr<int>(), cumsum_tensor.data_ptr<int>(),
      avg_length_of_neighbor_idxs, max_neighbour_distance, batch_size, M,
      nsample, neighbor_type);
  AT_CUDA_CHECK(cudaGetLastError());
}

void StackQueryThreeNNLocalIdxsCUDAKernelLauncher(
    const Tensor support_xyz_tensor, const Tensor new_xyz_tensor,
    const Tensor new_xyz_grid_centers_tensor, Tensor new_xyz_grid_idxs_tensor,
    Tensor new_xyz_grid_dist2_tensor, Tensor stack_neighbor_idxs_tensor,
    Tensor start_len_tensor, const int M, const int num_total_grids) {
  at::cuda::CUDAGuard device_guard(support_xyz_tensor.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(DIVUP(M, THREADS_PER_BLOCK), num_total_grids);
  dim3 threads(THREADS_PER_BLOCK);

  query_three_nn_by_stacked_local_idxs_cuda_kernel<<<blocks, threads>>>(
      support_xyz_tensor.data_ptr<float>(), new_xyz_tensor.data_ptr<float>(),
      new_xyz_grid_centers_tensor.data_ptr<float>(),
      new_xyz_grid_idxs_tensor.data_ptr<int>(),
      new_xyz_grid_dist2_tensor.data_ptr<float>(),
      stack_neighbor_idxs_tensor.data_ptr<int>(),
      start_len_tensor.data_ptr<int>(), M, num_total_grids);
  AT_CUDA_CHECK(cudaGetLastError());
}
