// Copyright (c) OpenMMLab. All rights reserved.
#include <iostream>

#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA

void CorrelationForwardCUDAKernelLauncher(Tensor input1, Tensor input2,
                                          Tensor output, int kH, int kW,
                                          int patchH, int patchW, int padH,
                                          int padW, int dilationH,
                                          int dilationW, int dilation_patchH,
                                          int dilation_patchW, int dH, int dW);

void CorrelationBackwardCUDAKernelLauncher(Tensor grad_output, Tensor input1,
                                           Tensor input2, Tensor grad_input1,
                                           Tensor grad_input2, int kH, int kW,
                                           int patchH, int patchW, int padH,
                                           int padW, int dilationH,
                                           int dilationW, int dilation_patchH,
                                           int dilation_patchW, int dH, int dW);

void correlation_cuda_forward(Tensor input1, Tensor input2, Tensor output,
                              int kH, int kW, int patchH, int patchW, int padH,
                              int padW, int dilationH, int dilationW,
                              int dilation_patchH, int dilation_patchW, int dH,
                              int dW) {
  CorrelationForwardCUDAKernelLauncher(
      input1, input2, output, kH, kW, patchH, patchW, padH, padW, dilationH,
      dilationW, dilation_patchH, dilation_patchW, dH, dW);
}

void correlation_cuda_backward(Tensor grad_output, Tensor input1, Tensor input2,
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

#endif

void correlation_forward(Tensor input1, Tensor input2, Tensor output, int kH,
                         int kW, int patchH, int patchW, int padH, int padW,
                         int dilationH, int dilationW, int dilation_patchH,
                         int dilation_patchW, int dH, int dW) {
  if (input1.device().is_cuda() && input2.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input1);
    CHECK_CUDA_INPUT(input2);
    correlation_cuda_forward(input1, input2, output, kH, kW, patchH, patchW,
                             padH, padW, dilationH, dilationW, dilation_patchH,
                             dilation_patchW, dH, dW);
#else
    AT_ERROR("Correlation is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("Correlation is not implemented on CPU");
  }
}

void correlation_backward(Tensor grad_output, Tensor input1, Tensor input2,
                          Tensor grad_input1, Tensor grad_input2, int kH,
                          int kW, int patchH, int patchW, int padH, int padW,
                          int dilationH, int dilationW, int dilation_patchH,
                          int dilation_patchW, int dH, int dW) {
  if (input1.device().is_cuda() && input2.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(grad_output);
    CHECK_CUDA_INPUT(input1);
    CHECK_CUDA_INPUT(input2);
    correlation_cuda_backward(grad_output, input1, input2, grad_input1,
                              grad_input2, kH, kW, patchH, patchW, padH, padW,
                              dilationH, dilationW, dilation_patchH,
                              dilation_patchW, dH, dW);

#else
    AT_ERROR("Correlation is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("Correlation is not implemented on CPU");
  }
}
