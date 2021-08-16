// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void BorderAlignForwardCUDAKernelLauncher(const Tensor &input,
                                          const Tensor &boxes, Tensor output,
                                          Tensor argmax_idx,
                                          const int pool_size);

void BorderAlignBackwardCUDAKernelLauncher(const Tensor &grad_output,
                                           const Tensor &boxes,
                                           const Tensor &argmax_idx,
                                           Tensor grad_input,
                                           const int pool_size);

void border_align_forward_cuda(const Tensor &input, const Tensor &boxes,
                               Tensor output, Tensor argmax_idx,
                               const int pool_size) {
  BorderAlignForwardCUDAKernelLauncher(input, boxes, output, argmax_idx,
                                       pool_size);
}

void border_align_backward_cuda(const Tensor &grad_output, const Tensor &boxes,
                                const Tensor &argmax_idx, Tensor grad_input,
                                const int pool_size) {
  BorderAlignBackwardCUDAKernelLauncher(grad_output, boxes, argmax_idx,
                                        grad_input, pool_size);
}
#endif

void border_align_forward(const Tensor &input, const Tensor &boxes,
                          Tensor output, Tensor argmax_idx,
                          const int pool_size) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(boxes);
    CHECK_CUDA_INPUT(output);
    CHECK_CUDA_INPUT(argmax_idx);

    border_align_forward_cuda(input, boxes, output, argmax_idx, pool_size);
#else
    AT_ERROR("BorderAlign is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("BorderAlign is not implemented on CPU");
  }
}

void border_align_backward(const Tensor &grad_output, const Tensor &boxes,
                           const Tensor &argmax_idx, Tensor grad_input,
                           const int pool_size) {
  if (grad_output.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(grad_output);
    CHECK_CUDA_INPUT(boxes);
    CHECK_CUDA_INPUT(argmax_idx);
    CHECK_CUDA_INPUT(grad_input);

    border_align_backward_cuda(grad_output, boxes, argmax_idx, grad_input,
                               pool_size);
#else
    AT_ERROR("BorderAlign is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("BorderAlign is not implemented on CPU");
  }
}
