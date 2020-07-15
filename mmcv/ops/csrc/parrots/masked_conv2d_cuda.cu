#include "masked_conv2d_cuda_kernel.cuh"
#include "parrots_cuda_helper.hpp"

void MaskedIm2colForwardCUDAKernelLauncher(
    const DArrayLite bottom_data, const DArrayLite mask_h_idx,
    const DArrayLite mask_w_idx, DArrayLite top_data, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, cudaStream_t stream) {
  int channels = bottom_data.dim(1);
  int height = bottom_data.dim(2);
  int width = bottom_data.dim(3);
  int mask_cnt = mask_h_idx.dim(0);
  int output_size = mask_cnt * channels;

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      bottom_data.elemType().prim(), ([&] {
        MaskedIm2colForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, bottom_data.ptr<scalar_t>(), height, width,
                kernel_h, kernel_w, pad_h, pad_w, mask_h_idx.ptr<int64_t>(),
                mask_w_idx.ptr<int64_t>(), mask_cnt, top_data.ptr<scalar_t>());
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void MaskedCol2imForwardCUDAKernelLaucher(const DArrayLite bottom_data,
                                          const DArrayLite mask_h_idx,
                                          const DArrayLite mask_w_idx,
                                          DArrayLite top_data, const int height,
                                          const int width, const int channels,
                                          cudaStream_t stream) {
  int mask_cnt = mask_h_idx.dim(0);
  int output_size = mask_cnt * channels;

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      bottom_data.elemType().prim(), ([&] {
        MaskedCol2imForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, bottom_data.ptr<scalar_t>(), height, width,
                channels, mask_h_idx.ptr<int64_t>(), mask_w_idx.ptr<int64_t>(),
                mask_cnt, top_data.ptr<scalar_t>());
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}
