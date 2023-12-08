// Copyright (c) OpenMMLab. All rights reserved
#include "border_align_musa_kernel.muh"
#include "pytorch_musa_helper.hpp"

void BorderAlignForwardMUSAKernelLauncher(const Tensor &input,
                                          const Tensor &boxes, Tensor output,
                                          Tensor argmax_idx,
                                          const int pool_size) {
  // shape assertion
  AT_ASSERTM(input.ndimension() == 4,
             "non-empty 4D(batch mode) tensor expected for input feature");
  AT_ASSERTM(boxes.ndimension() == 3,
             "boxes must be 3D tensor with size of [B, H*W, 4]");

  int batch_size = input.size(0);
  int feat_channels = input.size(1);
  int channels = feat_channels / 4;
  int height = input.size(2);
  int width = input.size(3);
  // shape [N, box_size, 4] for boxes. (x1, y1, x2, y2) format
  int box_size = boxes.size(1);
  // shape [N, channels, box_size, 4] for output
  int nthreads = batch_size * channels * box_size;

  at::musa::MUSAGuard device_guard(input.device());
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  dim3 block(128, 4);
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "border_align_forward_musa_kernel", [&] {
        border_align_forward_musa_kernel<scalar_t>
            <<<GET_BLOCKS(nthreads), block, 0, stream>>>(
                nthreads, input.data_ptr<scalar_t>(),
                boxes.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                argmax_idx.data_ptr<int>(), channels, box_size, height, width,
                pool_size);
      });

  AT_MUSA_CHECK(musaGetLastError());
}

void BorderAlignBackwardMUSAKernelLauncher(const Tensor &grad_output,
                                           const Tensor &boxes,
                                           const Tensor &argmax_idx,
                                           Tensor grad_input,
                                           const int pool_size) {
  int batch_size = grad_input.size(0);
  int feat_channels = grad_input.size(1);
  int channels = feat_channels / 4;
  int height = grad_input.size(2);
  int width = grad_input.size(3);
  int box_size = boxes.size(1);
  int nthreads = batch_size * channels * box_size;

  at::musa::MUSAGuard device_guard(grad_output.device());
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  dim3 block(128, 4);
  AT_DISPATCH_FLOATING_TYPES(
      grad_output.scalar_type(), "border_align_backward_musa_kernel", [&] {
        border_align_backward_musa_kernel<scalar_t>
            <<<GET_BLOCKS(nthreads), block, 0, stream>>>(
                nthreads, grad_output.data_ptr<scalar_t>(),
                boxes.data_ptr<scalar_t>(), argmax_idx.data_ptr<int>(),
                grad_input.data_ptr<scalar_t>(), channels, box_size, height,
                width, pool_size);
      });

  AT_MUSA_CHECK(musaGetLastError());
}
