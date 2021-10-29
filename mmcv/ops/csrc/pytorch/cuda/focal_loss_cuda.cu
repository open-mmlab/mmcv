// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cuda_helper.hpp"
#include "pytorch_device_registry.hpp"
#include "sigmoid_focal_loss_cuda_kernel.cuh"
#include "softmax_focal_loss_cuda_kernel.cuh"

void SigmoidFocalLossForwardCUDAKernelLauncher(Tensor input, Tensor target,
                                               Tensor weight, Tensor output,
                                               const float gamma,
                                               const float alpha) {
  int output_size = output.numel();
  int num_classes = input.size(1);
  AT_ASSERTM(target.max().item<int64_t>() <= (int64_t)num_classes,
             "target label should smaller or equal than num classes");
  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "sigmoid_focal_loss_forward_cuda_kernel", [&] {
        sigmoid_focal_loss_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(), gamma, alpha, num_classes);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void SigmoidFocalLossBackwardCUDAKernelLauncher(Tensor input, Tensor target,
                                                Tensor weight,
                                                Tensor grad_input,
                                                const float gamma,
                                                const float alpha) {
  int output_size = grad_input.numel();
  int num_classes = input.size(1);

  at::cuda::CUDAGuard device_guard(grad_input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "sigmoid_focal_loss_backward_cuda_kernel", [&] {
        sigmoid_focal_loss_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
                grad_input.data_ptr<scalar_t>(), gamma, alpha, num_classes);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void SoftmaxFocalLossForwardCUDAKernelLauncher(Tensor softmax, Tensor target,
                                               Tensor weight, Tensor output,
                                               const float gamma,
                                               const float alpha) {
  int output_size = output.numel();
  int num_classes = softmax.size(1);

  AT_ASSERTM(target.max().item<int64_t>() <= (int64_t)num_classes,
             "target label should smaller or equal than num classes");
  at::cuda::CUDAGuard device_guard(softmax.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      softmax.scalar_type(), "softmax_focal_loss_forward_cuda_kernel", [&] {
        softmax_focal_loss_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, softmax.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(), gamma, alpha, num_classes);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void SoftmaxFocalLossBackwardCUDAKernelLauncher(Tensor softmax, Tensor target,
                                                Tensor weight, Tensor buff,
                                                Tensor grad_input,
                                                const float gamma,
                                                const float alpha) {
  int num_classes = softmax.size(1);

  int output_size = buff.numel();
  at::cuda::CUDAGuard device_guard(grad_input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_input.scalar_type(),
      "softmax_focal_loss_backward_cuda1_"
      "kernel",
      [&] {
        softmax_focal_loss_backward_cuda1_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, softmax.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
                buff.data_ptr<scalar_t>(), gamma, alpha, num_classes);
      });

  AT_CUDA_CHECK(cudaGetLastError());

  output_size = grad_input.numel();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_input.scalar_type(),
      "softmax_focal_loss_backward_cuda2_"
      "kernel",
      [&] {
        softmax_focal_loss_backward_cuda2_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, softmax.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), buff.data_ptr<scalar_t>(),
                grad_input.data_ptr<scalar_t>(), num_classes);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void sigmoid_focal_loss_forward_cuda(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha) {
  SigmoidFocalLossForwardCUDAKernelLauncher(input, target, weight, output,
                                            gamma, alpha);
}

void sigmoid_focal_loss_backward_cuda(Tensor input, Tensor target,
                                      Tensor weight, Tensor grad_input,
                                      float gamma, float alpha) {
  SigmoidFocalLossBackwardCUDAKernelLauncher(input, target, weight, grad_input,
                                             gamma, alpha);
}

void softmax_focal_loss_forward_cuda(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha) {
  SoftmaxFocalLossForwardCUDAKernelLauncher(input, target, weight, output,
                                            gamma, alpha);
}

void softmax_focal_loss_backward_cuda(Tensor input, Tensor target,
                                      Tensor weight, Tensor buff,
                                      Tensor grad_input, float gamma,
                                      float alpha) {
  SoftmaxFocalLossBackwardCUDAKernelLauncher(input, target, weight, buff,
                                             grad_input, gamma, alpha);
}

void sigmoid_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha);

void sigmoid_focal_loss_backward_impl(Tensor input, Tensor target,
                                      Tensor weight, Tensor grad_input,
                                      float gamma, float alpha);

void softmax_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha);

void softmax_focal_loss_backward_impl(Tensor input, Tensor target,
                                      Tensor weight, Tensor buff,
                                      Tensor grad_input, float gamma,
                                      float alpha);

REGISTER_DEVICE_IMPL(sigmoid_focal_loss_forward_impl, CUDA,
                     sigmoid_focal_loss_forward_cuda);
REGISTER_DEVICE_IMPL(sigmoid_focal_loss_backward_impl, CUDA,
                     sigmoid_focal_loss_backward_cuda);
REGISTER_DEVICE_IMPL(softmax_focal_loss_forward_impl, CUDA,
                     softmax_focal_loss_forward_cuda);
REGISTER_DEVICE_IMPL(softmax_focal_loss_backward_impl, CUDA,
                     softmax_focal_loss_backward_cuda);
