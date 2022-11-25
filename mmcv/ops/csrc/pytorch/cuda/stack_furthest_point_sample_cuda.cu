// Copyright (c) OpenMMLab. All rights reserved.
#include <torch/types.h>

#include "pytorch_cuda_helper.hpp"
#include "stack_furthest_point_sample_cuda_kernel.cuh"

void StackFurthestPointSamplingForwardCUDALauncher(Tensor points_tensor,
                                            Tensor temp_tensor,
                                            Tensor xyz_batch_cnt_tensor,
                                               Tensor idx_tensor,
                                               Tensor num_sampled_points_tensor){
  at::cuda::CUDAGuard device_guard(points_tensor.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int N = points_tensor.size(0);
  int batch_size = xyz_batch_cnt_tensor.size(0);
    stack_farthest_point_sample_cuda_kernel<1024><<<batch_size, 1024>>>(
    batch_size, N, points_tensor.data_ptr<float>(), temp_tensor.data_ptr<float>(), xyz_batch_cnt_tensor.data_ptr<int>(), idx_tensor.data_ptr<int>(),
     num_sampled_points_tensor.data_ptr<int>()
    );

  AT_CUDA_CHECK(cudaGetLastError());
}
