// Copyright (c) 2018, SenseTime.
#include "parrots_cpp_helper.hpp"

void SigmoidFocalLossForwardCUDAKernelLauncher(
    const DArrayLite input, const DArrayLite target, const DArrayLite weight,
    DArrayLite output, float gamma, float alpha, cudaStream_t stream);

void SigmoidFocalLossBackwardCUDAKernelLauncher(
    const DArrayLite input, const DArrayLite target, const DArrayLite weight,
    DArrayLite grad_input, float gamma, float alpha, cudaStream_t stream);

void SoftmaxFocalLossForwardCUDAKernelLauncher(
    const DArrayLite input, const DArrayLite target, const DArrayLite weight,
    DArrayLite output, float gamma, float alpha, cudaStream_t stream);

void SoftmaxFocalLossBackwardCUDAKernelLauncher(
    const DArrayLite input, const DArrayLite target, const DArrayLite weight,
    DArrayLite buff, DArrayLite grad_input, float gamma, float alpha,
    cudaStream_t stream);

void sigmoid_focal_loss_forward_cuda(CudaContext& ctx, const SSElement& attr,
                                     const OperatorBase::in_list_t& ins,
                                     OperatorBase::out_list_t& outs) {
  float gamma;
  float alpha;
  SSAttrs(attr).get<float>("gamma", gamma).get<float>("alpha", alpha).done();

  // get inputs and outputs
  const auto& input = ins[0];
  const auto& target = ins[1];
  const auto& weight = ins[2];

  auto& output = outs[0];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());

  SigmoidFocalLossForwardCUDAKernelLauncher(input, target, weight, output,
                                            gamma, alpha, stream);
}

void sigmoid_focal_loss_backward_cuda(CudaContext& ctx, const SSElement& attr,
                                      const OperatorBase::in_list_t& ins,
                                      OperatorBase::out_list_t& outs) {
  float gamma;
  float alpha;
  SSAttrs(attr).get<float>("gamma", gamma).get<float>("alpha", alpha).done();

  // get inputs and outputs
  const auto& input = ins[0];
  const auto& target = ins[1];
  const auto& weight = ins[2];

  auto& grad_input = outs[0];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  SigmoidFocalLossBackwardCUDAKernelLauncher(input, target, weight, grad_input,
                                             gamma, alpha, stream);
}

void softmax_focal_loss_forward_cuda(CudaContext& ctx, const SSElement& attr,
                                     const OperatorBase::in_list_t& ins,
                                     OperatorBase::out_list_t& outs) {
  float gamma;
  float alpha;
  SSAttrs(attr).get<float>("gamma", gamma).get<float>("alpha", alpha).done();

  // get inputs and outputs
  const auto& input = ins[0];
  const auto& target = ins[1];
  const auto& weight = ins[2];

  auto& grad_input = outs[0];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());

  SoftmaxFocalLossForwardCUDAKernelLauncher(input, target, weight, grad_input,
                                            gamma, alpha, stream);
}

void softmax_focal_loss_backward_cuda(CudaContext& ctx, const SSElement& attr,
                                      const OperatorBase::in_list_t& ins,
                                      OperatorBase::out_list_t& outs) {
  float gamma;
  float alpha;
  SSAttrs(attr).get<float>("gamma", gamma).get<float>("alpha", alpha).done();

  // get inputs and outputs
  const auto& input = ins[0];
  const auto& target = ins[1];
  const auto& weight = ins[2];

  auto& buff = outs[0];
  auto& grad_input = outs[1];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  SoftmaxFocalLossBackwardCUDAKernelLauncher(input, target, weight, buff,
                                             grad_input, gamma, alpha, stream);
}

PARROTS_EXTENSION_REGISTER(sigmoid_focal_loss_forward)
    .attr("gamma")
    .attr("alpha")
    .input(3)
    .output(1)
    .apply(sigmoid_focal_loss_forward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(sigmoid_focal_loss_backward)
    .attr("gamma")
    .attr("alpha")
    .input(3)
    .output(1)
    .apply(sigmoid_focal_loss_backward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(softmax_focal_loss_forward)
    .attr("gamma")
    .attr("alpha")
    .input(3)
    .output(1)
    .apply(softmax_focal_loss_forward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(softmax_focal_loss_backward)
    .attr("gamma")
    .attr("alpha")
    .input(3)
    .output(2)
    .apply(softmax_focal_loss_backward_cuda)
    .done();
