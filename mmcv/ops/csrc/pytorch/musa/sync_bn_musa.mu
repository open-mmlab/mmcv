// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_musa_helper.hpp"
#include "sync_bn_musa_kernel.muh"

void SyncBNForwardMeanMUSAKernelLauncher(const Tensor input, Tensor mean) {
  int num = input.size(0);
  int channels = input.size(1);
  int spatial = input.size(2);

  at::musa::MUSAGuard device_guard(input.device());
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "sync_bn_forward_mean_musa_kernel", [&] {
        sync_bn_forward_mean_musa_kernel<scalar_t>
            <<<channels, THREADS_PER_BLOCK, 0, stream>>>(
                input.data_ptr<scalar_t>(), mean.data_ptr<float>(), num,
                channels, spatial);
      });
  AT_MUSA_CHECK(musaGetLastError());
}

void SyncBNForwardVarMUSAKernelLauncher(const Tensor input, const Tensor mean,
                                        Tensor var) {
  int num = input.size(0);
  int channels = input.size(1);
  int spatial = input.size(2);

  at::musa::MUSAGuard device_guard(input.device());
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "sync_bn_forward_mean_musa_kernel", [&] {
        sync_bn_forward_var_musa_kernel<scalar_t>
            <<<channels, THREADS_PER_BLOCK, 0, stream>>>(
                input.data_ptr<scalar_t>(), mean.data_ptr<float>(),
                var.data_ptr<float>(), num, channels, spatial);
      });
  AT_MUSA_CHECK(musaGetLastError());
}

void SyncBNForwardOutputMUSAKernelLauncher(
    const Tensor input, const Tensor mean, const Tensor var,
    Tensor running_mean, Tensor running_var, const Tensor weight,
    const Tensor bias, Tensor norm, Tensor std, Tensor output, float eps,
    float momentum, int group_size) {
  int num = input.size(0);
  int channels = input.size(1);
  int spatial = input.size(2);

  at::musa::MUSAGuard device_guard(input.device());
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "sync_bn_forward_mean_musa_kernel", [&] {
        sync_bn_forward_output_musa_kernel<scalar_t>
            <<<channels, THREADS_PER_BLOCK, 0, stream>>>(
                input.data_ptr<scalar_t>(), mean.data_ptr<float>(),
                var.data_ptr<float>(), running_mean.data_ptr<float>(),
                running_var.data_ptr<float>(), weight.data_ptr<float>(),
                bias.data_ptr<float>(), norm.data_ptr<float>(),
                std.data_ptr<float>(), output.data_ptr<scalar_t>(), num,
                channels, spatial, eps, momentum, group_size);
      });
  AT_MUSA_CHECK(musaGetLastError());
}

void SyncBNBackwardParamMUSAKernelLauncher(const Tensor grad_output,
                                           const Tensor norm,
                                           Tensor grad_weight,
                                           Tensor grad_bias) {
  int num = grad_output.size(0);
  int channels = grad_output.size(1);
  int spatial = grad_output.size(2);

  at::musa::MUSAGuard device_guard(grad_output.device());
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES(
      grad_output.scalar_type(), "sync_bn_backward_param_musa_kernel", [&] {
        sync_bn_backward_param_musa_kernel<scalar_t>
            <<<channels, THREADS_PER_BLOCK, 0, stream>>>(
                grad_output.data_ptr<scalar_t>(), norm.data_ptr<float>(),
                grad_weight.data_ptr<float>(), grad_bias.data_ptr<float>(), num,
                channels, spatial);
      });
  AT_MUSA_CHECK(musaGetLastError());
}

void SyncBNBackwardDataMUSAKernelLauncher(const Tensor grad_output,
                                          const Tensor weight,
                                          const Tensor grad_weight,
                                          const Tensor grad_bias,
                                          const Tensor norm, const Tensor std,
                                          Tensor grad_input) {
  int output_size = grad_input.numel();
  int num = grad_input.size(0);
  int channels = grad_input.size(1);
  int spatial = grad_input.size(2);

  at::musa::MUSAGuard device_guard(grad_input.device());
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES(
      grad_output.scalar_type(), "sync_bn_backward_data_musa_kernel", [&] {
        sync_bn_backward_data_musa_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, grad_output.data_ptr<scalar_t>(),
                weight.data_ptr<float>(), grad_weight.data_ptr<float>(),
                grad_bias.data_ptr<float>(), norm.data_ptr<float>(),
                std.data_ptr<float>(), grad_input.data_ptr<scalar_t>(), num,
                channels, spatial);
      });
  AT_MUSA_CHECK(musaGetLastError());
}
