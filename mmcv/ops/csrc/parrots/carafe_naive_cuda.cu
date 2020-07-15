#include "carafe_naive_cuda_kernel.cuh"
#include "parrots_cuda_helper.hpp"

void CARAFENAIVEForwardCUDAKernelLauncher(
    const DArrayLite features, const DArrayLite masks, DArrayLite output,
    const int kernel_size, const int group_size, const int scale_factor,
    cudaStream_t stream) {
  int output_size = output.size();
  int channels = output.dim(1);
  int height = output.dim(2);
  int width = output.dim(3);

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.elemType().prim(), ([&] {
        carafe_naive_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, features.ptr<scalar_t>(), masks.ptr<scalar_t>(),
                output.ptr<scalar_t>(), kernel_size, group_size, scale_factor,
                channels, height, width);
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void CARAFENAIVEBackwardCUDAKernelLauncher(
    const DArrayLite top_grad, const DArrayLite features,
    const DArrayLite masks, DArrayLite bottom_grad, DArrayLite mask_grad,
    const int kernel_size, const int group_size, const int scale_factor,
    cudaStream_t stream) {
  int output_size = top_grad.size();
  int channels = top_grad.dim(1);
  int height = top_grad.dim(2);
  int width = top_grad.dim(3);

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.elemType().prim(), ([&] {
        carafe_naive_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, top_grad.ptr<scalar_t>(), features.ptr<scalar_t>(),
                masks.ptr<scalar_t>(), bottom_grad.ptr<scalar_t>(),
                mask_grad.ptr<scalar_t>(), kernel_size, group_size,
                scale_factor, channels, height, width);
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}
