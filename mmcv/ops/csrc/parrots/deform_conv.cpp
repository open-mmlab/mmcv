// Copyright (c) 2018, SenseTime.
#include "parrots_cpp_helper.hpp"

void DeformConvForwardCUDAKernelLauncher(
    const DArrayLite input, const DArrayLite weight, const DArrayLite offset,
    DArrayLite output, DArrayLite columns, DArrayLite ones, int kW, int kH,
    int dW, int dH, int padW, int padH, int dilationW, int dilationH, int group,
    int deformable_group, int im2col_step, CudaContext& ctx,
    cudaStream_t stream);

void DeformConvBackwardInputCUDAKernelLauncher(
    const DArrayLite input, const DArrayLite offset,
    const DArrayLite gradOutput, DArrayLite gradInput, DArrayLite gradOffset,
    DArrayLite weight, DArrayLite columns, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationW, int dilationH, int group,
    int deformable_group, int im2col_step, CudaContext& ctx,
    cudaStream_t stream);

void DeformConvBackwardParametersCUDAKernelLauncher(
    const DArrayLite input, const DArrayLite offset,
    const DArrayLite gradOutput, DArrayLite gradWeight, DArrayLite columns,
    DArrayLite ones, int kW, int kH, int dW, int dH, int padW, int padH,
    int dilationW, int dilationH, int group, int deformable_group, float scale,
    int im2col_step, CudaContext& ctx, cudaStream_t stream);

void deform_conv_forward_cuda(CudaContext& ctx, const SSElement& attr,
                              const OperatorBase::in_list_t& ins,
                              OperatorBase::out_list_t& outs) {
  int kW, kH, dW, dH, padW, padH, dilationW, dilationH, group, deformable_group,
      im2col_step;
  SSAttrs(attr)
      .get<int>("kW", kW)
      .get<int>("kH", kH)
      .get<int>("dW", dW)
      .get<int>("dH", dH)
      .get<int>("padW", padW)
      .get<int>("padH", padH)
      .get<int>("dilationW", dilationW)
      .get<int>("dilationH", dilationH)
      .get<int>("group", group)
      .get<int>("deformable_group", deformable_group)
      .get<int>("im2col_step", im2col_step)
      .done();

  const auto input = ins[0];
  const auto weight = ins[1];
  const auto offset = ins[2];

  auto output = outs[0];
  auto columns = outs[1];
  auto ones = outs[2];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  DeformConvForwardCUDAKernelLauncher(
      input, weight, offset, output, columns, ones, kW, kH, dW, dH, padW, padH,
      dilationW, dilationH, group, deformable_group, im2col_step, ctx, stream);
}

void deform_conv_backward_input_cuda(CudaContext& ctx, const SSElement& attr,
                                     const OperatorBase::in_list_t& ins,
                                     OperatorBase::out_list_t& outs) {
  int kW, kH, dW, dH, padW, padH, dilationW, dilationH, group, deformable_group,
      im2col_step;
  SSAttrs(attr)
      .get<int>("kW", kW)
      .get<int>("kH", kH)
      .get<int>("dW", dW)
      .get<int>("dH", dH)
      .get<int>("padW", padW)
      .get<int>("padH", padH)
      .get<int>("dilationW", dilationW)
      .get<int>("dilationH", dilationH)
      .get<int>("group", group)
      .get<int>("deformable_group", deformable_group)
      .get<int>("im2col_step", im2col_step)
      .done();

  auto input = ins[0];
  auto offset = ins[1];
  auto gradOutput = ins[2];

  auto gradInput = outs[0];
  auto gradOffset = outs[1];
  auto weight = outs[2];
  auto columns = outs[3];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  DeformConvBackwardInputCUDAKernelLauncher(
      input, offset, gradOutput, gradInput, gradOffset, weight, columns, kW, kH,
      dW, dH, padW, padH, dilationW, dilationH, group, deformable_group,
      im2col_step, ctx, stream);
}

void deform_conv_backward_parameters_cuda(CudaContext& ctx,
                                          const SSElement& attr,
                                          const OperatorBase::in_list_t& ins,
                                          OperatorBase::out_list_t& outs) {
  int kW, kH, dW, dH, padW, padH, dilationW, dilationH, group, deformable_group,
      im2col_step;
  float scale;
  SSAttrs(attr)
      .get<int>("kW", kW)
      .get<int>("kH", kH)
      .get<int>("dW", dW)
      .get<int>("dH", dH)
      .get<int>("padW", padW)
      .get<int>("padH", padH)
      .get<int>("dilationW", dilationW)
      .get<int>("dilationH", dilationH)
      .get<int>("group", group)
      .get<int>("deformable_group", deformable_group)
      .get<float>("scale", scale)
      .get<int>("im2col_step", im2col_step)
      .done();

  auto input = ins[0];
  auto offset = ins[1];
  auto gradOutput = ins[2];

  auto gradWeight = outs[0];
  auto columns = outs[1];
  auto ones = outs[2];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  DeformConvBackwardParametersCUDAKernelLauncher(
      input, offset, gradOutput, gradWeight, columns, ones, kW, kH, dW, dH,
      padW, padH, dilationW, dilationH, group, deformable_group, scale,
      im2col_step, ctx, stream);
}

PARROTS_EXTENSION_REGISTER(deform_conv_forward)
    .attr("kW")
    .attr("kH")
    .attr("dW")
    .attr("dH")
    .attr("padW")
    .attr("padH")
    .attr("dilationW")
    .attr("dilationH")
    .attr("group")
    .attr("deformable_group")
    .attr("im2col_step")
    .input(3)
    .output(3)
    .apply(deform_conv_forward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(deform_conv_backward_input)
    .attr("kW")
    .attr("kH")
    .attr("dW")
    .attr("dH")
    .attr("padW")
    .attr("padH")
    .attr("dilationW")
    .attr("dilationH")
    .attr("group")
    .attr("deformable_group")
    .attr("im2col_step")
    .input(3)
    .output(4)
    .apply(deform_conv_backward_input_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(deform_conv_backward_parameters)
    .attr("kW")
    .attr("kH")
    .attr("dW")
    .attr("dH")
    .attr("padW")
    .attr("padH")
    .attr("dilationW")
    .attr("dilationH")
    .attr("group")
    .attr("deformable_group")
    .attr("scale")
    .attr("im2col_step")
    .input(3)
    .output(3)
    .apply(deform_conv_backward_parameters_cuda)
    .done();
