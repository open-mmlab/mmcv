#include "parrots_cuda_helper.hpp"
#include "tin_shift_cuda_kernel.cuh"

void TINShiftForwardCUDAKernelLauncher(const DArrayLite input,
                                       const DArrayLite shift,
                                       DArrayLite output, cudaStream_t stream) {
  int output_size = output.size();
  int batch_size = input.dim(0);
  int t_size = input.dim(1);
  int channels = input.dim(2);
  int hw_size = input.dim(3);
  int group_size = shift.dim(1);
  int group_channel = channels / group_size;
  int num_kernels = batch_size * hw_size * channels;

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.elemType().prim(), ([&] {
        tin_shift_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.ptr<scalar_t>(), shift.ptr<int>(),
                output.ptr<scalar_t>(), batch_size, channels, t_size, hw_size,
                group_size, group_channel);
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void TINShiftBackwardCUDAKernelLauncher(const DArrayLite grad_output,
                                        const DArrayLite shift,
                                        DArrayLite grad_input,
                                        cudaStream_t stream) {
  int output_size = grad_output.size();
  int batch_size = grad_output.dim(0);
  int t_size = grad_output.dim(1);
  int channels = grad_output.dim(2);
  int hw_size = grad_output.dim(3);
  int group_size = shift.dim(1);
  int group_channel = channels / group_size;
  int num_kernels = batch_size * hw_size * channels;

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.elemType().prim(), ([&] {
        tin_shift_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, grad_output.ptr<scalar_t>(), shift.ptr<int>(),
                grad_input.ptr<scalar_t>(), batch_size, channels, t_size,
                hw_size, group_size, group_channel);
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}
