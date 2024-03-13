// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_musa_helper.hpp"
#include "pytorch_device_registry.hpp"
#include "tin_shift_musa_kernel.muh"

void TINShiftForwardMUSAKernelLauncher(Tensor input, Tensor shift,
                                       Tensor output) {
  int output_size = output.numel();
  int batch_size = input.size(0);
  int t_size = input.size(1);
  int channels = input.size(2);
  int hw_size = input.size(3);
  int group_size = shift.size(1);
  int group_channel = channels / group_size;
  int num_kernels = batch_size * hw_size * channels;

  c10::musa::MUSAGuard device_guard(input.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "tin_shift_forward_musa_kernel", [&] {
        tin_shift_forward_musa_kernel<scalar_t>
            <<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.data_ptr<scalar_t>(), shift.data_ptr<int>(),
                output.data_ptr<scalar_t>(), batch_size, channels, t_size,
                hw_size, group_size, group_channel);
      });

  AT_MUSA_CHECK(musaGetLastError());
}

void TINShiftBackwardMUSAKernelLauncher(Tensor grad_output, Tensor shift,
                                        Tensor grad_input) {
  int output_size = grad_output.numel();
  int batch_size = grad_output.size(0);
  int t_size = grad_output.size(1);
  int channels = grad_output.size(2);
  int hw_size = grad_output.size(3);
  int group_size = shift.size(1);
  int group_channel = channels / group_size;
  int num_kernels = batch_size * hw_size * channels;

  c10::musa::MUSAGuard device_guard(grad_output.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES(
      grad_output.scalar_type(), "tin_shift_backward_musa_kernel", [&] {
        tin_shift_backward_musa_kernel<scalar_t>
            <<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, grad_output.data_ptr<scalar_t>(),
                shift.data_ptr<int>(), grad_input.data_ptr<scalar_t>(),
                batch_size, channels, t_size, hw_size, group_size,
                group_channel);
      });

  AT_MUSA_CHECK(musaGetLastError());
}
