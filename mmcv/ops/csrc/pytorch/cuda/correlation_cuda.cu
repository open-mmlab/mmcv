// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/ClementPinard/Pytorch-Correlation-extension/blob/master/Correlation_Module/correlation_cuda_kernel.cu
// Original licence: Under MIT License

#include "correlation_cuda.cuh"
#include "pytorch_cuda_helper.hpp"
#include "pytorch_device_registry.hpp"

void CorrelationForwardCUDAKernelLauncher(Tensor input1, Tensor input2,
                                          Tensor output, int kH, int kW,
                                          int patchH, int patchW, int padH,
                                          int padW, int dilationH,
                                          int dilationW, int dilation_patchH,
                                          int dilation_patchW, int dH, int dW) {
  const int batch_size = input1.size(0);
  const int iH = input1.size(2);
  const int iW = input1.size(3);
  const int dilatedKH = (kH - 1) * dilationH + 1;
  const int dilatedKW = (kW - 1) * dilationW + 1;

  const auto oH = (iH + 2 * padH - dilatedKH) / dH + 1;
  const auto oW = (iW + 2 * padW - dilatedKW) / dW + 1;

  auto trInput1 = input1.permute({0, 2, 3, 1}).contiguous();
  auto trInput2 = input2.permute({0, 2, 3, 1}).contiguous();

  const int threads = THREADS_FORWARD;
  const dim3 blocks(batch_size, oH, oW);

  at::cuda::CUDAGuard device_guard(input1.device());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input1.scalar_type(), "correlation_forward_cuda", ([&] {
        TensorAcc4R trInput1_acc =
            trInput1.packed_accessor32<scalar_t, 4, RestrictPtrTraits>();
        TensorAcc4R trInput2_acc =
            trInput2.packed_accessor32<scalar_t, 4, RestrictPtrTraits>();
        TensorAcc5R output_acc =
            output.packed_accessor32<scalar_t, 5, RestrictPtrTraits>();

        correlation_forward_cuda_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                trInput1_acc, trInput2_acc, output_acc, kH, kW, patchH, patchW,
                padH, padW, dilationH, dilationW, dilation_patchH,
                dilation_patchW, dH, dW);
      }));
}

void CorrelationBackwardCUDAKernelLauncher(
    Tensor grad_output, Tensor input1, Tensor input2, Tensor grad_input1,
    Tensor grad_input2, int kH, int kW, int patchH, int patchW, int padH,
    int padW, int dilationH, int dilationW, int dilation_patchH,
    int dilation_patchW, int dH, int dW) {
  const int batch_size = input1.size(0);
  const int iH = input1.size(2);
  const int iW = input1.size(3);
  const int C = input1.size(1);

  const dim3 blocks(C, iH, iW);
  const dim3 threads(THREADS_BACKWARD, THREADS_BACKWARD);

  at::cuda::CUDAGuard device_guard(input1.device());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input1.scalar_type(), "correlation_backward_cuda", ([&] {
        TensorAcc4R input1_acc =
            input1.packed_accessor32<scalar_t, 4, RestrictPtrTraits>();
        TensorAcc4R input2_acc =
            input2.packed_accessor32<scalar_t, 4, RestrictPtrTraits>();
        TensorAcc4R grad_input1_acc =
            grad_input1.packed_accessor32<scalar_t, 4, RestrictPtrTraits>();
        TensorAcc4R grad_input2_acc =
            grad_input2.packed_accessor32<scalar_t, 4, RestrictPtrTraits>();
        TensorAcc5R grad_output_acc =
            grad_output.packed_accessor32<scalar_t, 5, RestrictPtrTraits>();

        for (int n = 0; n < batch_size; ++n) {
          correlation_backward_cuda_kernel_input1<scalar_t>
              <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                  grad_output_acc, input2_acc, grad_input1_acc, kH, kW, patchH,
                  patchW, padH, padW, dilationH, dilationW, dilation_patchH,
                  dilation_patchW, dH, dW, n);
        }

        for (int n = 0; n < batch_size; ++n) {
          correlation_backward_cuda_kernel_input2<scalar_t>
              <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                  grad_output_acc, input1_acc, grad_input2_acc, kH, kW, patchH,
                  patchW, padH, padW, dilationH, dilationW, dilation_patchH,
                  dilation_patchW, dH, dW, n);
        }
      }));
}

void correlation_forward_cuda(Tensor input1, Tensor input2, Tensor output,
                              int kH, int kW, int patchH, int patchW, int padH,
                              int padW, int dilationH, int dilationW,
                              int dilation_patchH, int dilation_patchW, int dH,
                              int dW) {
  CorrelationForwardCUDAKernelLauncher(
      input1, input2, output, kH, kW, patchH, patchW, padH, padW, dilationH,
      dilationW, dilation_patchH, dilation_patchW, dH, dW);
}

void correlation_backward_cuda(Tensor grad_output, Tensor input1, Tensor input2,
                               Tensor grad_input1, Tensor grad_input2, int kH,
                               int kW, int patchH, int patchW, int padH,
                               int padW, int dilationH, int dilationW,
                               int dilation_patchH, int dilation_patchW, int dH,
                               int dW) {
  CorrelationBackwardCUDAKernelLauncher(
      grad_output, input1, input2, grad_input1, grad_input2, kH, kW, patchH,
      patchW, padH, padW, dilationH, dilationW, dilation_patchH,
      dilation_patchW, dH, dW);
}

void correlation_forward_impl(Tensor input1, Tensor input2, Tensor output,
                              int kH, int kW, int patchH, int patchW, int padH,
                              int padW, int dilationH, int dilationW,
                              int dilation_patchH, int dilation_patchW, int dH,
                              int dW);

void correlation_backward_impl(Tensor grad_output, Tensor input1, Tensor input2,
                               Tensor grad_input1, Tensor grad_input2, int kH,
                               int kW, int patchH, int patchW, int padH,
                               int padW, int dilationH, int dilationW,
                               int dilation_patchH, int dilation_patchW, int dH,
                               int dW);

REGISTER_DEVICE_IMPL(correlation_forward_impl, CUDA, correlation_forward_cuda);
REGISTER_DEVICE_IMPL(correlation_backward_impl, CUDA,
                     correlation_backward_cuda);
