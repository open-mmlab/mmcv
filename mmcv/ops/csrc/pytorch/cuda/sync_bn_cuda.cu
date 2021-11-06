// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cuda_helper.hpp"
#include "pytorch_device_registry.hpp"
#include "sync_bn_cuda_kernel.cuh"

void SyncBNForwardMeanCUDAKernelLauncher(const Tensor input, Tensor mean) {
  int num = input.size(0);
  int channels = input.size(1);
  int spatial = input.size(2);

  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "sync_bn_forward_mean_cuda_kernel", [&] {
        sync_bn_forward_mean_cuda_kernel<scalar_t>
            <<<channels, THREADS_PER_BLOCK, 0, stream>>>(
                input.data_ptr<scalar_t>(), mean.data_ptr<float>(), num,
                channels, spatial);
      });
  AT_CUDA_CHECK(cudaGetLastError());
}

void SyncBNForwardVarCUDAKernelLauncher(const Tensor input, const Tensor mean,
                                        Tensor var) {
  int num = input.size(0);
  int channels = input.size(1);
  int spatial = input.size(2);

  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "sync_bn_forward_mean_cuda_kernel", [&] {
        sync_bn_forward_var_cuda_kernel<scalar_t>
            <<<channels, THREADS_PER_BLOCK, 0, stream>>>(
                input.data_ptr<scalar_t>(), mean.data_ptr<float>(),
                var.data_ptr<float>(), num, channels, spatial);
      });
  AT_CUDA_CHECK(cudaGetLastError());
}

void SyncBNForwardOutputCUDAKernelLauncher(
    const Tensor input, const Tensor mean, const Tensor var,
    Tensor running_mean, Tensor running_var, const Tensor weight,
    const Tensor bias, Tensor norm, Tensor std, Tensor output, float eps,
    float momentum, int group_size) {
  int num = input.size(0);
  int channels = input.size(1);
  int spatial = input.size(2);

  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "sync_bn_forward_mean_cuda_kernel", [&] {
        sync_bn_forward_output_cuda_kernel<scalar_t>
            <<<channels, THREADS_PER_BLOCK, 0, stream>>>(
                input.data_ptr<scalar_t>(), mean.data_ptr<float>(),
                var.data_ptr<float>(), running_mean.data_ptr<float>(),
                running_var.data_ptr<float>(), weight.data_ptr<float>(),
                bias.data_ptr<float>(), norm.data_ptr<float>(),
                std.data_ptr<float>(), output.data_ptr<scalar_t>(), num,
                channels, spatial, eps, momentum, group_size);
      });
  AT_CUDA_CHECK(cudaGetLastError());
}

void SyncBNBackwardParamCUDAKernelLauncher(const Tensor grad_output,
                                           const Tensor norm,
                                           Tensor grad_weight,
                                           Tensor grad_bias) {
  int num = grad_output.size(0);
  int channels = grad_output.size(1);
  int spatial = grad_output.size(2);

  at::cuda::CUDAGuard device_guard(grad_output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "sync_bn_backward_param_cuda_kernel", [&] {
        sync_bn_backward_param_cuda_kernel<scalar_t>
            <<<channels, THREADS_PER_BLOCK, 0, stream>>>(
                grad_output.data_ptr<scalar_t>(), norm.data_ptr<float>(),
                grad_weight.data_ptr<float>(), grad_bias.data_ptr<float>(), num,
                channels, spatial);
      });
  AT_CUDA_CHECK(cudaGetLastError());
}

void SyncBNBackwardDataCUDAKernelLauncher(const Tensor grad_output,
                                          const Tensor weight,
                                          const Tensor grad_weight,
                                          const Tensor grad_bias,
                                          const Tensor norm, const Tensor std,
                                          Tensor grad_input) {
  int output_size = grad_input.numel();
  int num = grad_input.size(0);
  int channels = grad_input.size(1);
  int spatial = grad_input.size(2);

  at::cuda::CUDAGuard device_guard(grad_input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "sync_bn_backward_data_cuda_kernel", [&] {
        sync_bn_backward_data_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, grad_output.data_ptr<scalar_t>(),
                weight.data_ptr<float>(), grad_weight.data_ptr<float>(),
                grad_bias.data_ptr<float>(), norm.data_ptr<float>(),
                std.data_ptr<float>(), grad_input.data_ptr<scalar_t>(), num,
                channels, spatial);
      });
  AT_CUDA_CHECK(cudaGetLastError());
}

void sync_bn_forward_mean_cuda(const Tensor input, Tensor mean) {
  SyncBNForwardMeanCUDAKernelLauncher(input, mean);
}

void sync_bn_forward_var_cuda(const Tensor input, const Tensor mean,
                              Tensor var) {
  SyncBNForwardVarCUDAKernelLauncher(input, mean, var);
}

void sync_bn_forward_output_cuda(const Tensor input, const Tensor mean,
                                 const Tensor var, Tensor running_mean,
                                 Tensor running_var, const Tensor weight,
                                 const Tensor bias, Tensor norm, Tensor std,
                                 Tensor output, float eps, float momentum,
                                 int group_size) {
  SyncBNForwardOutputCUDAKernelLauncher(input, mean, var, running_mean,
                                        running_var, weight, bias, norm, std,
                                        output, eps, momentum, group_size);
}

void sync_bn_backward_param_cuda(const Tensor grad_output, const Tensor norm,
                                 Tensor grad_weight, Tensor grad_bias) {
  SyncBNBackwardParamCUDAKernelLauncher(grad_output, norm, grad_weight,
                                        grad_bias);
}

void sync_bn_backward_data_cuda(const Tensor grad_output, const Tensor weight,
                                const Tensor grad_weight,
                                const Tensor grad_bias, const Tensor norm,
                                const Tensor std, Tensor grad_input) {
  SyncBNBackwardDataCUDAKernelLauncher(grad_output, weight, grad_weight,
                                       grad_bias, norm, std, grad_input);
}

void sync_bn_forward_mean_impl(const Tensor input, Tensor mean);

void sync_bn_forward_var_impl(const Tensor input, const Tensor mean,
                              Tensor var);

void sync_bn_forward_output_impl(const Tensor input, const Tensor mean,
                                 const Tensor var, Tensor running_mean,
                                 Tensor running_var, const Tensor weight,
                                 const Tensor bias, Tensor norm, Tensor std,
                                 Tensor output, float eps, float momentum,
                                 int group_size);

void sync_bn_backward_param_impl(const Tensor grad_output, const Tensor norm,
                                 Tensor grad_weight, Tensor grad_bias);

void sync_bn_backward_data_impl(const Tensor grad_output, const Tensor weight,
                                const Tensor grad_weight,
                                const Tensor grad_bias, const Tensor norm,
                                const Tensor std, Tensor grad_input);

REGISTER_DEVICE_IMPL(sync_bn_forward_mean_impl, CUDA,
                     sync_bn_forward_mean_cuda);
REGISTER_DEVICE_IMPL(sync_bn_forward_var_impl, CUDA, sync_bn_forward_var_cuda);
REGISTER_DEVICE_IMPL(sync_bn_forward_output_impl, CUDA,
                     sync_bn_forward_output_cuda);
REGISTER_DEVICE_IMPL(sync_bn_backward_param_impl, CUDA,
                     sync_bn_backward_param_cuda);
REGISTER_DEVICE_IMPL(sync_bn_backward_data_impl, CUDA,
                     sync_bn_backward_data_cuda);
