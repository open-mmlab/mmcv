// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/csuhan/s2anet/blob/master/mmdet/ops/orn/src/musa/ActiveRotatingFilter_musa.cu
#include "active_rotated_filter_musa_kernel.muh"
#include "pytorch_musa_helper.hpp"

void ActiveRotatedFilterForwardMUSAKernelLauncher(const Tensor input,
                                                  const Tensor indices,
                                                  Tensor output) {
  int num_output_planes = input.size(0);
  int num_input_planes = input.size(1);
  int num_orientations = input.size(2);
  int kH = input.size(3);
  int kW = input.size(4);
  int num_rotations = indices.size(3);
  int nEntry = num_orientations * kH * kW;
  int output_size = input.numel();

  c10::musa::MUSAGuard device_guard(input.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "active_rotated_filter_forward_musa_kernel", [&] {
        active_rotated_filter_forward_musa_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.data_ptr<scalar_t>(),
                indices.data_ptr<int>(), num_input_planes, num_output_planes,
                num_orientations, num_rotations, nEntry,
                output.data_ptr<scalar_t>());
      });
  AT_MUSA_CHECK(musaGetLastError());
}

void ActiveRotatedFilterBackwardMUSAKernelLauncher(const Tensor grad_out,
                                                   const Tensor indices,
                                                   Tensor grad_in) {
  int num_orientations = indices.size(0);
  int kH = indices.size(1);
  int kW = indices.size(2);
  int num_rotations = indices.size(3);
  int num_output_planes = grad_out.size(0) / num_rotations;
  int num_input_planes = grad_out.size(1) / num_orientations;
  int nEntry = num_orientations * kH * kW;
  int output_size = grad_in.numel();

  c10::musa::MUSAGuard device_guard(indices.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES(
      grad_out.scalar_type(), "active_rotated_filter_backward_musa_kernel",
      [&] {
        active_rotated_filter_backward_musa_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, grad_out.data_ptr<scalar_t>(),
                indices.data_ptr<int>(), num_input_planes, num_output_planes,
                num_orientations, num_rotations, nEntry,
                grad_in.data_ptr<scalar_t>());
      });
  AT_MUSA_CHECK(musaGetLastError());
}
