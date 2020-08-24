#include "deform_roi_pool_cuda_kernel.cuh"
#include "parrots_cuda_helper.hpp"

void DeformRoIPoolForwardCUDAKernelLauncher(
    const DArrayLite input, const DArrayLite rois, const DArrayLite offset,
    DArrayLite output, int pooled_height, int pooled_width, float spatial_scale,
    int sampling_ratio, float gamma, cudaStream_t stream) {
  int output_size = output.size();
  int channels = input.dim(1);
  int height = input.dim(2);
  int width = input.dim(3);

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.elemType().prim(), ([&] {
        deform_roi_pool_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.ptr<scalar_t>(), rois.ptr<scalar_t>(),
                offset.ptr<scalar_t>(), output.ptr<scalar_t>(), pooled_height,
                pooled_width, spatial_scale, sampling_ratio, gamma, channels,
                height, width);
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void DeformRoIPoolBackwardCUDAKernelLauncher(
    const DArrayLite grad_output, const DArrayLite input, const DArrayLite rois,
    const DArrayLite offset, DArrayLite grad_input, DArrayLite grad_offset,
    int pooled_height, int pooled_width, float spatial_scale,
    int sampling_ratio, float gamma, cudaStream_t stream) {
  int output_size = grad_output.size();
  int channels = grad_input.dim(1);
  int height = grad_input.dim(2);
  int width = grad_input.dim(3);

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.elemType().prim(), ([&] {
        deform_roi_pool_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, grad_output.ptr<scalar_t>(), input.ptr<scalar_t>(),
                rois.ptr<scalar_t>(), offset.ptr<scalar_t>(),
                grad_input.ptr<scalar_t>(), grad_offset.ptr<scalar_t>(),
                pooled_height, pooled_width, spatial_scale, sampling_ratio,
                gamma, channels, height, width);
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}
