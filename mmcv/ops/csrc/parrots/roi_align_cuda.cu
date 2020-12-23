#include "parrots_cuda_helper.hpp"
#include "roi_align_cuda_kernel.cuh"

void ROIAlignForwardCUDAKernelLauncher(DArrayLite input, DArrayLite rois,
                                       DArrayLite output, DArrayLite argmax_y,
                                       DArrayLite argmax_x, int aligned_height,
                                       int aligned_width, float spatial_scale,
                                       int sampling_ratio, int pool_mode,
                                       bool aligned, cudaStream_t stream) {
  int output_size = output.size();
  int channels = input.dim(1);
  int height = input.dim(2);
  int width = input.dim(3);

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.elemType().prim(), ([&] {
        roi_align_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.ptr<scalar_t>(), rois.ptr<scalar_t>(),
                output.ptr<scalar_t>(), argmax_y.ptr<scalar_t>(),
                argmax_x.ptr<scalar_t>(), aligned_height, aligned_width,
                static_cast<scalar_t>(spatial_scale), sampling_ratio, pool_mode,
                aligned, channels, height, width);
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void ROIAlignBackwardCUDAKernelLauncher(
    DArrayLite grad_output, DArrayLite rois, DArrayLite argmax_y,
    DArrayLite argmax_x, DArrayLite grad_input, int aligned_height,
    int aligned_width, float spatial_scale, int sampling_ratio, int pool_mode,
    bool aligned, cudaStream_t stream) {
  int output_size = grad_output.size();
  int channels = grad_input.dim(1);
  int height = grad_input.dim(2);
  int width = grad_input.dim(3);

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.elemType().prim(), ([&] {
        roi_align_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, grad_output.ptr<scalar_t>(), rois.ptr<scalar_t>(),
                argmax_y.ptr<scalar_t>(), argmax_x.ptr<scalar_t>(),
                grad_input.ptr<scalar_t>(), aligned_height, aligned_width,
                static_cast<scalar_t>(spatial_scale), sampling_ratio, pool_mode,
                aligned, channels, height, width);
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}
