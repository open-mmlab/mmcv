// Copyright (c) OpenMMLab. All rights reserved
// Modified from
// https://github.com/hszhao/semseg/blob/master/lib/psa/src

#include <torch/serialize/tensor.h>

#include "psamask_musa_kernel.muh"
#include "pytorch_musa_helper.hpp"

void PSAMaskForwardMUSAKernelLauncher(const int psa_type, const Tensor input,
                                      Tensor output, const int num_,
                                      const int h_feature, const int w_feature,
                                      const int h_mask, const int w_mask,
                                      const int half_h_mask,
                                      const int half_w_mask) {
  int nthreads = num_ * h_feature * w_feature;
  musaStream_t stream = c10::musa::getCurrentMUSAStream();
  if (psa_type == 0)
    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "psamask_collect_forward_musa", [&] {
          psamask_collect_forward_musa<scalar_t><<<nthreads, 512, 0, stream>>>(
              nthreads, h_feature, w_feature, h_mask, w_mask, half_h_mask,
              half_w_mask, input.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>());
        });
  else
    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "psamask_distribute_forward_musa", [&] {
          psamask_distribute_forward_musa<scalar_t>
              <<<nthreads, 512, 0, stream>>>(
                  nthreads, h_feature, w_feature, h_mask, w_mask, half_h_mask,
                  half_w_mask, input.data_ptr<scalar_t>(),
                  output.data_ptr<scalar_t>());
        });
}

void PSAMaskBackwardMUSAKernelLauncher(
    const int psa_type, const Tensor grad_output, Tensor grad_input,
    const int num_, const int h_feature, const int w_feature, const int h_mask,
    const int w_mask, const int half_h_mask, const int half_w_mask) {
  int nthreads = num_ * h_feature * w_feature;
  musaStream_t stream = c10::musa::getCurrentMUSAStream();
  if (psa_type == 0)
    AT_DISPATCH_FLOATING_TYPES(
        grad_input.scalar_type(), "psamask_collect_backward_musa", [&] {
          psamask_collect_backward_musa<scalar_t><<<nthreads, 512, 0, stream>>>(
              nthreads, h_feature, w_feature, h_mask, w_mask, half_h_mask,
              half_w_mask, grad_output.data_ptr<scalar_t>(),
              grad_input.data_ptr<scalar_t>());
        });
  else
    AT_DISPATCH_FLOATING_TYPES(
        grad_input.scalar_type(), "psamask_distribute_backward_musa", [&] {
          psamask_distribute_backward_musa<scalar_t>
              <<<nthreads, 512, 0, stream>>>(
                  nthreads, h_feature, w_feature, h_mask, w_mask, half_h_mask,
                  half_w_mask, grad_output.data_ptr<scalar_t>(),
                  grad_input.data_ptr<scalar_t>());
        });
}
