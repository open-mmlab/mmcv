// Copyright (c) OpenMMLab. All rights reserved
#include "carafe_naive_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"
#include "pytorch_device_registry.hpp"

void CARAFENAIVEForwardCUDAKernelLauncher(const Tensor features,
                                          const Tensor masks, Tensor output,
                                          const int kernel_size,
                                          const int group_size,
                                          const int scale_factor) {
  int output_size = output.numel();
  int channels = output.size(1);
  int height = output.size(2);
  int width = output.size(3);

  at::cuda::CUDAGuard device_guard(features.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.scalar_type(), "CARAFENAIVEForward", ([&] {
        carafe_naive_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, features.data_ptr<scalar_t>(),
                masks.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                kernel_size, group_size, scale_factor, channels, height, width);
      }));

  AT_CUDA_CHECK(cudaGetLastError());
}

void CARAFENAIVEBackwardCUDAKernelLauncher(
    const Tensor top_grad, const Tensor features, const Tensor masks,
    Tensor bottom_grad, Tensor mask_grad, const int kernel_size,
    const int group_size, const int scale_factor) {
  int output_size = top_grad.numel();
  int channels = top_grad.size(1);
  int height = top_grad.size(2);
  int width = top_grad.size(3);

  at::cuda::CUDAGuard device_guard(top_grad.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.scalar_type(), "CARAFENAIVEBackward", ([&] {
        carafe_naive_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, top_grad.data_ptr<scalar_t>(),
                features.data_ptr<scalar_t>(), masks.data_ptr<scalar_t>(),
                bottom_grad.data_ptr<scalar_t>(),
                mask_grad.data_ptr<scalar_t>(), kernel_size, group_size,
                scale_factor, channels, height, width);
      }));

  AT_CUDA_CHECK(cudaGetLastError());
}

void carafe_naive_forward_cuda(Tensor features, Tensor masks, Tensor output,
                               int kernel_size, int group_size,
                               int scale_factor) {
  CARAFENAIVEForwardCUDAKernelLauncher(features, masks, output, kernel_size,
                                       group_size, scale_factor);
}

void carafe_naive_backward_cuda(Tensor top_grad, Tensor features, Tensor masks,
                                Tensor bottom_grad, Tensor mask_grad,
                                int kernel_size, int group_size,
                                int scale_factor) {
  CARAFENAIVEBackwardCUDAKernelLauncher(top_grad, features, masks, bottom_grad,
                                        mask_grad, kernel_size, group_size,
                                        scale_factor);
}
void carafe_naive_forward_impl(Tensor features, Tensor masks, Tensor output,
                               int kernel_size, int group_size,
                               int scale_factor);

void carafe_naive_backward_impl(Tensor top_grad, Tensor features, Tensor masks,
                                Tensor bottom_grad, Tensor mask_grad,
                                int kernel_size, int group_size,
                                int scale_factor);

REGISTER_DEVICE_IMPL(carafe_naive_forward_impl, CUDA,
                     carafe_naive_forward_cuda);
REGISTER_DEVICE_IMPL(carafe_naive_backward_impl, CUDA,
                     carafe_naive_backward_cuda);
