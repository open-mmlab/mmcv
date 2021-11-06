// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cuda_helper.hpp"
#include "pytorch_device_registry.hpp"
#include "tin_shift_cuda_kernel.cuh"

void TINShiftForwardCUDAKernelLauncher(Tensor input, Tensor shift,
                                       Tensor output) {
  int output_size = output.numel();
  int batch_size = input.size(0);
  int t_size = input.size(1);
  int channels = input.size(2);
  int hw_size = input.size(3);
  int group_size = shift.size(1);
  int group_channel = channels / group_size;
  int num_kernels = batch_size * hw_size * channels;

  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "tin_shift_forward_cuda_kernel", [&] {
        tin_shift_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.data_ptr<scalar_t>(), shift.data_ptr<int>(),
                output.data_ptr<scalar_t>(), batch_size, channels, t_size,
                hw_size, group_size, group_channel);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void TINShiftBackwardCUDAKernelLauncher(Tensor grad_output, Tensor shift,
                                        Tensor grad_input) {
  int output_size = grad_output.numel();
  int batch_size = grad_output.size(0);
  int t_size = grad_output.size(1);
  int channels = grad_output.size(2);
  int hw_size = grad_output.size(3);
  int group_size = shift.size(1);
  int group_channel = channels / group_size;
  int num_kernels = batch_size * hw_size * channels;

  at::cuda::CUDAGuard device_guard(grad_output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "tin_shift_backward_cuda_kernel", [&] {
        tin_shift_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, grad_output.data_ptr<scalar_t>(),
                shift.data_ptr<int>(), grad_input.data_ptr<scalar_t>(),
                batch_size, channels, t_size, hw_size, group_size,
                group_channel);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void tin_shift_forward_cuda(Tensor input, Tensor shift, Tensor output) {
  TINShiftForwardCUDAKernelLauncher(input, shift, output);
}

void tin_shift_backward_cuda(Tensor grad_output, Tensor shift,
                             Tensor grad_input) {
  TINShiftBackwardCUDAKernelLauncher(grad_output, shift, grad_input);
}

void tin_shift_forward_impl(Tensor input, Tensor shift, Tensor output);
void tin_shift_backward_impl(Tensor grad_output, Tensor shift,
                             Tensor grad_input);
REGISTER_DEVICE_IMPL(tin_shift_forward_impl, CUDA, tin_shift_forward_cuda);
REGISTER_DEVICE_IMPL(tin_shift_backward_impl, CUDA, tin_shift_backward_cuda);
