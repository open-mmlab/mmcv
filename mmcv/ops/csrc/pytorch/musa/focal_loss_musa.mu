// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_musa_helper.hpp"
#include "sigmoid_focal_loss_musa_kernel.muh"
#include "softmax_focal_loss_musa_kernel.muh"

void SigmoidFocalLossForwardMUSAKernelLauncher(Tensor input, Tensor target,
                                               Tensor weight, Tensor output,
                                               const float gamma,
                                               const float alpha) {
  int output_size = output.numel();
  int num_classes = input.size(1);
  AT_ASSERTM(target.max().item<int64_t>() <= (int64_t)num_classes,
             "target label should smaller or equal than num classes");
  c10::musa::MUSAGuard device_guard(input.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "sigmoid_focal_loss_forward_musa_kernel", [&] {
        sigmoid_focal_loss_forward_musa_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(), gamma, alpha, num_classes);
      });

  AT_MUSA_CHECK(musaGetLastError());
}

void SigmoidFocalLossBackwardMUSAKernelLauncher(Tensor input, Tensor target,
                                                Tensor weight,
                                                Tensor grad_input,
                                                const float gamma,
                                                const float alpha) {
  int output_size = grad_input.numel();
  int num_classes = input.size(1);

  c10::musa::MUSAGuard device_guard(grad_input.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "sigmoid_focal_loss_backward_musa_kernel", [&] {
        sigmoid_focal_loss_backward_musa_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
                grad_input.data_ptr<scalar_t>(), gamma, alpha, num_classes);
      });

  AT_MUSA_CHECK(musaGetLastError());
}

void SoftmaxFocalLossForwardMUSAKernelLauncher(Tensor softmax, Tensor target,
                                               Tensor weight, Tensor output,
                                               const float gamma,
                                               const float alpha) {
  int output_size = output.numel();
  int num_classes = softmax.size(1);

  AT_ASSERTM(target.max().item<int64_t>() <= (int64_t)num_classes,
             "target label should smaller or equal than num classes");
  c10::musa::MUSAGuard device_guard(softmax.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES(
      softmax.scalar_type(), "softmax_focal_loss_forward_musa_kernel", [&] {
        softmax_focal_loss_forward_musa_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, softmax.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(), gamma, alpha, num_classes);
      });

  AT_MUSA_CHECK(musaGetLastError());
}

void SoftmaxFocalLossBackwardMUSAKernelLauncher(Tensor softmax, Tensor target,
                                                Tensor weight, Tensor buff,
                                                Tensor grad_input,
                                                const float gamma,
                                                const float alpha) {
  int num_classes = softmax.size(1);

  int output_size = buff.numel();
  c10::musa::MUSAGuard device_guard(grad_input.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES(
      grad_input.scalar_type(),
      "softmax_focal_loss_backward_musa1_"
      "kernel",
      [&] {
        softmax_focal_loss_backward_musa1_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, softmax.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
                buff.data_ptr<scalar_t>(), gamma, alpha, num_classes);
      });

  AT_MUSA_CHECK(musaGetLastError());

  output_size = grad_input.numel();
  AT_DISPATCH_FLOATING_TYPES(
      grad_input.scalar_type(),
      "softmax_focal_loss_backward_musa2_"
      "kernel",
      [&] {
        softmax_focal_loss_backward_musa2_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, softmax.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), buff.data_ptr<scalar_t>(),
                grad_input.data_ptr<scalar_t>(), num_classes);
      });

  AT_MUSA_CHECK(musaGetLastError());
}
