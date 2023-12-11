// Copyright (c) OpenMMLab. All rights reserved
#include "bezier_align_musa_kernel.muh"
#include "pytorch_musa_helper.hpp"

void BezierAlignForwardMUSAKernelLauncher(Tensor input, Tensor rois,
                                          Tensor output, int aligned_height,
                                          int aligned_width,
                                          float spatial_scale,
                                          int sampling_ratio, bool aligned) {
  int output_size = output.numel();
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  c10::musa::MUSAGuard device_guard(input.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "bezier_align_forward_musa_kernel", [&] {
        bezier_align_forward_musa_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.data_ptr<scalar_t>(),
                rois.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                aligned_height, aligned_width,
                static_cast<scalar_t>(spatial_scale), sampling_ratio, aligned,
                channels, height, width);
      });

  AT_MUSA_CHECK(musaGetLastError());
}

void BezierAlignBackwardMUSAKernelLauncher(
    Tensor grad_output, Tensor rois, Tensor grad_input, int aligned_height,
    int aligned_width, float spatial_scale, int sampling_ratio, bool aligned) {
  int output_size = grad_output.numel();
  int channels = grad_input.size(1);
  int height = grad_input.size(2);
  int width = grad_input.size(3);

  c10::musa::MUSAGuard device_guard(grad_output.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES(
      grad_output.scalar_type(), "bezier_align_backward_musa_kernel", [&] {
        bezier_align_backward_musa_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, grad_output.data_ptr<scalar_t>(),
                rois.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>(),
                aligned_height, aligned_width,
                static_cast<scalar_t>(spatial_scale), sampling_ratio, aligned,
                channels, height, width);
      });

  AT_MUSA_CHECK(musaGetLastError());
}
