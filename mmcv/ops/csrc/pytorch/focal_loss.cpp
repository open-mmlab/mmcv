// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#ifdef MMCV_WITH_MLU
#include "pytorch_mlu_helper.hpp"
#endif

#ifdef MMCV_WITH_CUDA
void SigmoidFocalLossForwardCUDAKernelLauncher(Tensor input, Tensor target,
                                               Tensor weight, Tensor output,
                                               const float gamma,
                                               const float alpha);

void SigmoidFocalLossBackwardCUDAKernelLauncher(Tensor input, Tensor target,
                                                Tensor weight,
                                                Tensor grad_input,
                                                const float gamma,
                                                const float alpha);

void SoftmaxFocalLossForwardCUDAKernelLauncher(Tensor input, Tensor target,
                                               Tensor weight, Tensor output,
                                               const float gamma,
                                               const float alpha);

void SoftmaxFocalLossBackwardCUDAKernelLauncher(Tensor input, Tensor target,
                                                Tensor weight, Tensor buff,
                                                Tensor grad_input,
                                                const float gamma,
                                                const float alpha);

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
#endif

#ifdef MMCV_WITH_MLU
void SigmoidFocalLossForwardMLUKernelLauncher(Tensor input, Tensor target,
                                              Tensor weight, Tensor output,
                                              const float gamma,
                                              const float alpha);

void SigmoidFocalLossBackwardMLUKernelLauncher(Tensor input, Tensor target,
                                               Tensor weight, Tensor grad_input,
                                               const float gamma,
                                               const float alpha);

void sigmoid_focal_loss_forward_mlu(Tensor input, Tensor target, Tensor weight,
                                    Tensor output, float gamma, float alpha) {
  SigmoidFocalLossForwardMLUKernelLauncher(input, target, weight, output, gamma,
                                           alpha);
}

void sigmoid_focal_loss_backward_mlu(Tensor input, Tensor target, Tensor weight,
                                     Tensor grad_input, float gamma,
                                     float alpha) {
  SigmoidFocalLossBackwardMLUKernelLauncher(input, target, weight, grad_input,
                                            gamma, alpha);
}
#endif

void sigmoid_focal_loss_forward(Tensor input, Tensor target, Tensor weight,
                                Tensor output, float gamma, float alpha) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(target);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(output);

    sigmoid_focal_loss_forward_cuda(input, target, weight, output, gamma,
                                    alpha);
#else
    AT_ERROR("SigmoidFocalLoss is not compiled with GPU support");
#endif
#ifdef MMCV_WITH_MLU
  } else if (input.device().type() == at::kMLU) {
    CHECK_MLU_INPUT(input);
    CHECK_MLU_INPUT(target);
    CHECK_MLU_INPUT(weight);
    CHECK_MLU_INPUT(output);
    sigmoid_focal_loss_forward_mlu(input, target, weight, output, gamma, alpha);
#endif
  } else {
    AT_ERROR("SigmoidFocalLoss is not implemented on CPU");
  }
}

void sigmoid_focal_loss_backward(Tensor input, Tensor target, Tensor weight,
                                 Tensor grad_input, float gamma, float alpha) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(target);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(grad_input);

    sigmoid_focal_loss_backward_cuda(input, target, weight, grad_input, gamma,
                                     alpha);
#else
    AT_ERROR("SigmoidFocalLoss is not compiled with GPU support");
#endif
#ifdef MMCV_WITH_MLU
  } else if (input.device().type() == at::kMLU) {
    CHECK_MLU_INPUT(input);
    CHECK_MLU_INPUT(target);
    CHECK_MLU_INPUT(weight);
    CHECK_MLU_INPUT(grad_input);

    sigmoid_focal_loss_backward_mlu(input, target, weight, grad_input, gamma,
                                    alpha);
#endif
  } else {
    AT_ERROR("SigmoidFocalLoss is not implemented on CPU");
  }
}

void softmax_focal_loss_forward(Tensor input, Tensor target, Tensor weight,
                                Tensor output, float gamma, float alpha) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(target);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(output);

    softmax_focal_loss_forward_cuda(input, target, weight, output, gamma,
                                    alpha);
#else
    AT_ERROR("SoftmaxFocalLoss is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("SoftmaxFocalLoss is not implemented on CPU");
  }
}

void softmax_focal_loss_backward(Tensor input, Tensor target, Tensor weight,
                                 Tensor buff, Tensor grad_input, float gamma,
                                 float alpha) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(target);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(buff);
    CHECK_CUDA_INPUT(grad_input);

    softmax_focal_loss_backward_cuda(input, target, weight, buff, grad_input,
                                     gamma, alpha);
#else
    AT_ERROR("SoftmaxFocalLoss is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("SoftmaxFocalLoss is not implemented on CPU");
  }
}
