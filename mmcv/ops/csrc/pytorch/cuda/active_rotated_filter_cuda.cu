// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/csuhan/s2anet/blob/master/mmdet/ops/orn/src/cuda/ActiveRotatingFilter_cuda.cu
#include "active_rotated_filter_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

void ARFForwardLauncher(const Tensor input, const Tensor indices,
                        Tensor output) {
  int nOutputPlane = input.size(0);
  int nInputPlane = input.size(1);
  int num_orientations = input.size(2);
  int kH = input.size(3);
  int kW = input.size(4);
  int num_rotations = indices.size(3);
  int nEntry = num_orientations * kH * kW;
  int output_size = output.numel();

  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "ARF_forward", [&] {
    ARF_forward_cuda_kernel<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
            output_size, input.data_ptr<scalar_t>(),
            indices.data_ptr<int>(), nInputPlane, nOutputPlane,
            num_orientations, num_rotations, nEntry,
            output.data<scalar_t>());
  });
  AT_CUDA_CHECK(cudaGetLastError());
}

void ARFBackwardLauncher(const Tensor grad_out, const Tensor indices,
                         Tensor grad_in) {
  int num_orientations = indices.size(0);
  int kH = indices.size(1);
  int kW = indices.size(2);
  int num_rotations = indices.size(3);
  int nOutputPlane = grad_out.size(0) / num_rotations;
  int nInputPlane = grad_out.size(1) / num_orientations;
  int nEntry = num_orientations * kH * kW;
  int output_size = grad_in.numel();

  at::cuda::CUDAGuard device_guard(indices.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "ARF_backward", [&] {
    ARF_backward_cuda_kernel<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
            output_size, grad_out.data_ptr<scalar_t>(),
            indices.data_ptr<int>(), nInputPlane, nOutputPlane,
            num_orientations, num_rotations, nEntry,
            grad_in.data_ptr<scalar_t>());
  });
  AT_CUDA_CHECK(cudaGetLastError());
}
