#include "parrots_cuda_helper.hpp"
#include "sync_bn_cuda_kernel.cuh"

void SyncBNForwardMeanCUDAKernelLauncher(const DArrayLite input,
                                         DArrayLite mean, cudaStream_t stream) {
  int num = input.dim(0);
  int channels = input.dim(1);
  int spatial = input.dim(2);

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.elemType().prim(), ([&] {
        sync_bn_forward_mean_cuda_kernel<scalar_t>
            <<<channels, THREADS_PER_BLOCK, 0, stream>>>(input.ptr<scalar_t>(),
                                                         mean.ptr<float>(), num,
                                                         channels, spatial);
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void SyncBNForwardVarCUDAKernelLauncher(const DArrayLite input,
                                        const DArrayLite mean, DArrayLite var,
                                        cudaStream_t stream) {
  int num = input.dim(0);
  int channels = input.dim(1);
  int spatial = input.dim(2);

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.elemType().prim(), ([&] {
        sync_bn_forward_var_cuda_kernel<scalar_t>
            <<<channels, THREADS_PER_BLOCK, 0, stream>>>(
                input.ptr<scalar_t>(), mean.ptr<float>(), var.ptr<float>(), num,
                channels, spatial);
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void SyncBNForwardOutputCUDAKernelLauncher(
    const DArrayLite input, const DArrayLite mean, const DArrayLite var,
    DArrayLite running_mean, DArrayLite running_var, const DArrayLite weight,
    const DArrayLite bias, DArrayLite norm, DArrayLite std, DArrayLite output,
    float eps, float momentum, size_t group_size, cudaStream_t stream) {
  int num = input.dim(0);
  int channels = input.dim(1);
  int spatial = input.dim(2);

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.elemType().prim(), ([&] {
        sync_bn_forward_output_cuda_kernel<scalar_t>
            <<<channels, THREADS_PER_BLOCK, 0, stream>>>(
                input.ptr<scalar_t>(), mean.ptr<float>(), var.ptr<float>(),
                running_mean.ptr<float>(), running_var.ptr<float>(),
                weight.ptr<float>(), bias.ptr<float>(), norm.ptr<float>(),
                std.ptr<float>(), output.ptr<scalar_t>(), num, channels,
                spatial, eps, momentum, group_size);
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void SyncBNBackwardParamCUDAKernelLauncher(const DArrayLite grad_output,
                                           const DArrayLite norm,
                                           DArrayLite grad_weight,
                                           DArrayLite grad_bias,
                                           cudaStream_t stream) {
  int num = grad_output.dim(0);
  int channels = grad_output.dim(1);
  int spatial = grad_output.dim(2);

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.elemType().prim(), ([&] {
        sync_bn_backward_param_cuda_kernel<scalar_t>
            <<<channels, THREADS_PER_BLOCK, 0, stream>>>(
                grad_output.ptr<scalar_t>(), norm.ptr<float>(),
                grad_weight.ptr<float>(), grad_bias.ptr<float>(), num, channels,
                spatial);
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void SyncBNBackwardDataCUDAKernelLauncher(
    const DArrayLite grad_output, const DArrayLite weight,
    const DArrayLite grad_weight, const DArrayLite grad_bias,
    const DArrayLite norm, const DArrayLite std, DArrayLite grad_input,
    cudaStream_t stream) {
  int output_size = grad_input.size();
  int num = grad_input.dim(0);
  int channels = grad_input.dim(1);
  int spatial = grad_input.dim(2);

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_input.elemType().prim(), ([&] {
        sync_bn_backward_data_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, grad_output.ptr<scalar_t>(), weight.ptr<float>(),
                grad_weight.ptr<float>(), grad_bias.ptr<float>(),
                norm.ptr<float>(), std.ptr<float>(), grad_input.ptr<scalar_t>(),
                num, channels, spatial);
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}
