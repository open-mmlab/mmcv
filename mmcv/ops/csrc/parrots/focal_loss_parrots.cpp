// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "focal_loss_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void sigmoid_focal_loss_forward_cuda_parrots(CudaContext& ctx,
                                             const SSElement& attr,
                                             const OperatorBase::in_list_t& ins,
                                             OperatorBase::out_list_t& outs) {
  float gamma;
  float alpha;
  SSAttrs(attr).get<float>("gamma", gamma).get<float>("alpha", alpha).done();

  // get inputs and outputs
  const auto& input = buildATensor(ctx, ins[0]);
  const auto& target = buildATensor(ctx, ins[1]);
  const auto& weight = buildATensor(ctx, ins[2]);

  auto output = buildATensor(ctx, outs[0]);

  sigmoid_focal_loss_forward_cuda(input, target, weight, output, gamma, alpha);
}

void sigmoid_focal_loss_backward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  float gamma;
  float alpha;
  SSAttrs(attr).get<float>("gamma", gamma).get<float>("alpha", alpha).done();

  // get inputs and outputs
  const auto& input = buildATensor(ctx, ins[0]);
  const auto& target = buildATensor(ctx, ins[1]);
  const auto& weight = buildATensor(ctx, ins[2]);

  auto grad_input = buildATensor(ctx, outs[0]);

  sigmoid_focal_loss_backward_cuda(input, target, weight, grad_input, gamma,
                                   alpha);
}

void softmax_focal_loss_forward_cuda_parrots(CudaContext& ctx,
                                             const SSElement& attr,
                                             const OperatorBase::in_list_t& ins,
                                             OperatorBase::out_list_t& outs) {
  float gamma;
  float alpha;
  SSAttrs(attr).get<float>("gamma", gamma).get<float>("alpha", alpha).done();

  // get inputs and outputs
  const auto& input = buildATensor(ctx, ins[0]);
  const auto& target = buildATensor(ctx, ins[1]);
  const auto& weight = buildATensor(ctx, ins[2]);

  auto output = buildATensor(ctx, outs[0]);
  softmax_focal_loss_forward_cuda(input, target, weight, output, gamma, alpha);
}

void softmax_focal_loss_backward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  float gamma;
  float alpha;
  SSAttrs(attr).get<float>("gamma", gamma).get<float>("alpha", alpha).done();

  // get inputs and outputs
  const auto& input = buildATensor(ctx, ins[0]);
  const auto& target = buildATensor(ctx, ins[1]);
  const auto& weight = buildATensor(ctx, ins[2]);

  auto buff = buildATensor(ctx, outs[0]);
  auto grad_input = buildATensor(ctx, outs[1]);
  softmax_focal_loss_backward_cuda(input, target, weight, buff, grad_input,
                                   gamma, alpha);
}

PARROTS_EXTENSION_REGISTER(sigmoid_focal_loss_backward)
    .attr("gamma")
    .attr("alpha")
    .input(3)
    .output(1)
    .apply(sigmoid_focal_loss_backward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(softmax_focal_loss_forward)
    .attr("gamma")
    .attr("alpha")
    .input(3)
    .output(1)
    .apply(softmax_focal_loss_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(softmax_focal_loss_backward)
    .attr("gamma")
    .attr("alpha")
    .input(3)
    .output(2)
    .apply(softmax_focal_loss_backward_cuda_parrots)
    .done();
#endif

#ifdef PARROTS_USE_CAMB
#if 1
void sigmoidFocalLossForwardMLUKernelLauncher(CambContext &ctx,
                                              const DArrayLite &input, const DArrayLite &target,
                                              const DArrayLite &weight, DArrayLite &output, float gamma,
                                              float alpha);
#endif
void sigmoid_focal_loss_forward_camb_parrots(CambContext& ctx,
                                             const SSElement& attr,
                                             const OperatorBase::in_list_t& ins,
                                             OperatorBase::out_list_t& outs) {
    float gamma;
    float alpha;
    SSAttrs(attr).get<float>("gamma", gamma).get<float>("alpha", alpha).done();

    // get inputs and outputs
    const auto &input = ins[0];
    const auto &target = ins[1];
    const auto &weight = ins[2];
    auto &output = outs[0];

    sigmoidFocalLossForwardMLUKernelLauncher(
        ctx, input, target, weight, output, gamma, alpha);
}
#endif // PARROTS_USE_CAMB

PARROTS_EXTENSION_REGISTER(sigmoid_focal_loss_forward)
    .attr("gamma")
    .attr("alpha")
    .input(3)
    .output(1)
    #ifdef MMCV_WITH_CUDA
    .apply(sigmoid_focal_loss_forward_cuda_parrots)
    #endif // MMCV_WITH_CUDA
    #ifdef PARROTS_USE_CAMB
    .apply(sigmoid_focal_loss_forward_camb_parrots)
    #endif // PARROTS_USE_CAMB
    .done();