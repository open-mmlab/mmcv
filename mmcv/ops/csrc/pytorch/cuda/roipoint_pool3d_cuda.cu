/*
Modified from
https://github.com/sshaoshuai/PCDet/blob/master/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d_kernel.cu
Point cloud feature pooling
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/

#include <math.h>
#include <stdio.h>

#include "pytorch_cuda_helper.hpp"
#include "roipoint_pool3d_cuda_kernel.cuh"

void RoIPointPool3dForwardCUDAKernelLauncher(
    int batch_size, int pts_num, int boxes_num, int feature_in_len,
    int sampled_pts_num, const Tensor xyz, const Tensor boxes3d,
    const Tensor pts_feature, Tensor pooled_features,
    Tensor pooled_empty_flag) {
  int *pts_assign = NULL;
  cudaMalloc(&pts_assign, batch_size * pts_num * boxes_num *
                              sizeof(int));  // (batch_size, N, M)
  // cudaMemset(&pts_assign, -1, batch_size * pts_num * boxes_num *
  // sizeof(int));

  dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), boxes_num,
              batch_size);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      xyz.scalar_type(), "assign_pts_to_box3d", [&] {
        assign_pts_to_box3d<scalar_t><<<blocks, threads>>>(
            batch_size, pts_num, boxes_num, xyz.data_ptr<scalar_t>(),
            boxes3d.data_ptr<scalar_t>(), pts_assign);
      });

  int *pts_idx = NULL;
  cudaMalloc(&pts_idx, batch_size * boxes_num * sampled_pts_num *
                           sizeof(int));  // (batch_size, M, sampled_pts_num)

  dim3 blocks2(DIVUP(boxes_num, THREADS_PER_BLOCK),
               batch_size);  // blockIdx.x(col), blockIdx.y(row)

  get_pooled_idx<<<blocks2, threads>>>(batch_size, pts_num, boxes_num,
                                       sampled_pts_num, pts_assign, pts_idx,
                                       pooled_empty_flag.data_ptr<int>());

  dim3 blocks_pool(DIVUP(sampled_pts_num, THREADS_PER_BLOCK), boxes_num,
                   batch_size);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      xyz.scalar_type(), "roipoint_pool3d_forward", [&] {
        roipoint_pool3d_forward<scalar_t><<<blocks_pool, threads>>>(
            batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num,
            xyz.data_ptr<scalar_t>(), pts_idx, pts_feature.data_ptr<scalar_t>(),
            pooled_features.data_ptr<scalar_t>(),
            pooled_empty_flag.data_ptr<int>());
      });

  cudaFree(pts_assign);
  cudaFree(pts_idx);
}
