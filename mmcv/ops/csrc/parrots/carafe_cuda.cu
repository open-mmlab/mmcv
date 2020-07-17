#include "carafe_cuda_kernel.cuh"
#include "parrots_cuda_helper.hpp"

void CARAFEForwardCUDAKernelLauncher(
    const DArrayLite features, const DArrayLite masks, DArrayLite rfeatures,
    DArrayLite routput, DArrayLite rmasks, DArrayLite output,
    const int kernel_size, const int group_size, const int scale_factor,
    cudaStream_t stream) {
  const int batch_size = output.dim(0);
  const int channels = output.dim(1);
  const int output_height = output.dim(2);
  const int output_width = output.dim(3);

  const int input_height = features.dim(2);
  const int input_width = features.dim(3);

  const int mask_channels = masks.dim(1);

  // one warp per pixel
  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.elemType().prim(), ([&] {
        const int dh = divideUP(channels, kTileDim);
        const int dw = divideUP(input_height * input_width, kTileDim);
        BatchTranspose2DCUDAKernel<scalar_t>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, channels, input_height * input_width, dh, dw,
                features.ptr<scalar_t>(), rfeatures.ptr<scalar_t>());
      }));
  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.elemType().prim(), ([&] {
        const int dh = divideUP(mask_channels, kTileDim);
        const int dw = divideUP(output_height * output_width, kTileDim);
        BatchTranspose2DCUDAKernel<scalar_t>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, mask_channels, output_height * output_width, dh, dw,
                masks.ptr<scalar_t>(), rmasks.ptr<scalar_t>());
      }));
  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.elemType().prim(), ([&] {
        const int num_kernels =
            batch_size * output_height * output_width * THREADS_PER_PIXEL;

        CARAFEForward<scalar_t><<<divideUP(num_kernels, THREADS_PER_BLOCK),
                                  THREADS_PER_BLOCK, 0, stream>>>(
            num_kernels, rfeatures.ptr<scalar_t>(), rmasks.ptr<scalar_t>(),
            kernel_size, group_size, scale_factor, channels, input_height,
            input_width, output_height, output_width, mask_channels,
            routput.ptr<scalar_t>());
      }));
  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.elemType().prim(), ([&] {
        const int dh = divideUP(output_height * output_width, kTileDim);
        const int dw = divideUP(channels, kTileDim);
        BatchTranspose2DCUDAKernel<scalar_t>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, output_height * output_width, channels, dh, dw,
                routput.ptr<scalar_t>(), output.ptr<scalar_t>());
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void CARAFEBackwardCUDAKernelLauncher(
    const DArrayLite top_grad, const DArrayLite rfeatures,
    const DArrayLite masks, DArrayLite rtop_grad, DArrayLite rbottom_grad_hs,
    DArrayLite rbottom_grad, DArrayLite rmask_grad, DArrayLite bottom_grad,
    DArrayLite mask_grad, const int kernel_size, const int group_size,
    const int scale_factor, cudaStream_t stream) {
  const int batch_size = top_grad.dim(0);
  const int channels = top_grad.dim(1);
  const int output_height = top_grad.dim(2);
  const int output_width = top_grad.dim(3);

  const int input_height = bottom_grad.dim(2);
  const int input_width = bottom_grad.dim(3);

  const int mask_channels = masks.dim(1);

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.elemType().prim(), ([&] {
        const int dh = divideUP(channels, kTileDim);
        const int dw = divideUP(output_height * output_width, kTileDim);
        BatchTranspose2DCUDAKernel<scalar_t>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, channels, output_height * output_width, dh, dw,
                top_grad.ptr<scalar_t>(), rtop_grad.ptr<scalar_t>());
      }));
  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.elemType().prim(), ([&] {
        const int num_kernels =
            batch_size * output_height * output_width * THREADS_PER_PIXEL;

        CARAFEBackward_Feature<scalar_t>
            <<<divideUP(num_kernels, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0,
               stream>>>(num_kernels, rtop_grad.ptr<scalar_t>(),
                         masks.ptr<scalar_t>(), kernel_size, group_size,
                         scale_factor, channels, input_height, input_width,
                         output_height, output_width, mask_channels,
                         rbottom_grad_hs.ptr<scalar_t>());
      }));
  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.elemType().prim(), ([&] {
        const int num_kernels =
            batch_size * input_height * input_width * THREADS_PER_PIXEL;

        FeatureSum<scalar_t><<<divideUP(num_kernels, THREADS_PER_BLOCK),
                               THREADS_PER_BLOCK, 0, stream>>>(
            num_kernels, rbottom_grad_hs.ptr<scalar_t>(), scale_factor,
            channels, input_height, input_width, rbottom_grad.ptr<scalar_t>());
      }));
  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.elemType().prim(), ([&] {
        const int dh = divideUP(input_height * input_width, kTileDim);
        const int dw = divideUP(channels, kTileDim);
        BatchTranspose2DCUDAKernel<scalar_t>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, input_height * input_width, channels, dh, dw,
                rbottom_grad.ptr<scalar_t>(), bottom_grad.ptr<scalar_t>());
      }));
  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.elemType().prim(), ([&] {
        const int num_kernels = batch_size * output_height * output_width *
                                mask_channels * WARP_SIZE;

        CARAFEBackward_Mask<scalar_t>
            <<<divideUP(num_kernels, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0,
               stream>>>(num_kernels, rtop_grad.ptr<scalar_t>(),
                         rfeatures.ptr<scalar_t>(), kernel_size, group_size,
                         scale_factor, channels, input_height, input_width,
                         output_height, output_width, mask_channels,
                         rmask_grad.ptr<scalar_t>());
      }));
  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.elemType().prim(), ([&] {
        const int dh = divideUP(output_height * output_width, kTileDim);
        const int dw = divideUP(mask_channels, kTileDim);
        BatchTranspose2DCUDAKernel<scalar_t>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, output_height * output_width, mask_channels, dh, dw,
                rmask_grad.ptr<scalar_t>(), mask_grad.ptr<scalar_t>());
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}
