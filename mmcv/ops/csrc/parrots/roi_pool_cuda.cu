#include "parrots_cuda_helper.hpp"
#include "roi_pool_cuda_kernel.cuh"

void ROIPoolForwardCUDAKernelLauncher(const DArrayLite input,
                                      const DArrayLite rois, DArrayLite output,
                                      DArrayLite argmax, int pooled_height,
                                      int pooled_width, float spatial_scale,
                                      cudaStream_t stream) {
  int output_size = output.size();
  int channels = input.dim(1);
  int height = input.dim(2);
  int width = input.dim(3);

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(input.elemType().prim(), [&] {
    roi_pool_forward_cuda_kernel<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
            output_size, input.ptr<scalar_t>(), rois.ptr<scalar_t>(),
            output.ptr<scalar_t>(), argmax.ptr<int>(), pooled_height,
            pooled_width, spatial_scale, channels, height, width);
  });

  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void ROIPoolBackwardCUDAKernelLauncher(const DArrayLite grad_output,
                                       const DArrayLite rois,
                                       const DArrayLite argmax,
                                       DArrayLite grad_input, int pooled_height,
                                       int pooled_width, float spatial_scale,
                                       cudaStream_t stream) {
  int output_size = grad_output.size();
  int channels = grad_output.dim(1);
  int height = grad_output.dim(2);
  int width = grad_output.dim(3);

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.elemType().prim(), [&] {
    roi_pool_backward_cuda_kernel<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
            output_size, grad_output.ptr<scalar_t>(), rois.ptr<scalar_t>(),
            argmax.ptr<int>(), grad_input.ptr<scalar_t>(), pooled_height,
            pooled_width, channels, height, width);
  });

  PARROTS_CUDA_CHECK(cudaGetLastError());
}
