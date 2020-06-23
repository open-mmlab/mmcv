// Copyright (c) 2019, SenseTime.
#include "parrots_cpp_helper.hpp"

void ModulatedDeformConvForwardCUDAKernelLauncher(
    const DArrayLite input, const DArrayLite weight, const DArrayLite bias,
    const DArrayLite ones, const DArrayLite offset, const DArrayLite mask,
    DArrayLite output, DArrayLite columns, int kernel_h, int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, const int group,
    int deformable_group, const bool with_bias, CudaContext& ctx,
    cudaStream_t stream);

void ModulatedDeformConvBackwardCUDAKernelLauncher(
    const DArrayLite input, const DArrayLite weight, const DArrayLite bias,
    const DArrayLite ones, const DArrayLite offset, const DArrayLite mask,
    DArrayLite columns, DArrayLite grad_input, DArrayLite grad_weight,
    DArrayLite grad_bias, DArrayLite grad_offset, DArrayLite grad_mask,
    DArrayLite grad_output, int kernel_h, int kernel_w, int stride_h,
    int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w,
    int group, int deformable_group, const bool with_bias, CudaContext& ctx,
    cudaStream_t stream);

void modulated_deform_conv_forward_cuda(CudaContext& ctx, const SSElement& attr,
                                        const OperatorBase::in_list_t& ins,
                                        OperatorBase::out_list_t& outs) {
  int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
      dilation_w, group, deformable_group, with_bias;
  SSAttrs(attr)
      .get<int>("kernel_h", kernel_h)
      .get<int>("kernel_w", kernel_w)
      .get<int>("stride_h", stride_h)
      .get<int>("stride_w", stride_w)
      .get<int>("pad_h", pad_h)
      .get<int>("pad_w", pad_w)
      .get<int>("dilation_h", dilation_h)
      .get<int>("dilation_w", dilation_w)
      .get<int>("group", group)
      .get<int>("deformable_group", deformable_group)
      .get<int>("with_bias", with_bias)
      .done();

  auto input = ins[0];
  auto weight = ins[1];
  auto bias = ins[2];
  auto ones = ins[3];
  auto offset = ins[4];
  auto mask = ins[5];

  auto output = outs[0];
  auto columns = outs[1];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  ModulatedDeformConvForwardCUDAKernelLauncher(
      input, weight, bias, ones, offset, mask, output, columns, kernel_h,
      kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
      deformable_group, with_bias, ctx, stream);
}

void modulated_deform_conv_backward_cuda(CudaContext& ctx,
                                         const SSElement& attr,
                                         const OperatorBase::in_list_t& ins,
                                         OperatorBase::out_list_t& outs) {
  int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
      dilation_w, group, deformable_group, with_bias;
  SSAttrs(attr)
      .get<int>("kernel_h", kernel_h)
      .get<int>("kernel_w", kernel_w)
      .get<int>("stride_h", stride_h)
      .get<int>("stride_w", stride_w)
      .get<int>("pad_h", pad_h)
      .get<int>("pad_w", pad_w)
      .get<int>("dilation_h", dilation_h)
      .get<int>("dilation_w", dilation_w)
      .get<int>("group", group)
      .get<int>("deformable_group", deformable_group)
      .get<int>("with_bias", with_bias)
      .done();

  auto input = ins[0];
  auto weight = ins[1];
  auto bias = ins[2];
  auto ones = ins[3];
  auto offset = ins[4];
  auto mask = ins[5];

  auto columns = outs[0];
  auto grad_input = outs[1];
  auto grad_weight = outs[2];
  auto grad_bias = outs[3];
  auto grad_offset = outs[4];
  auto grad_mask = outs[5];
  auto grad_output = outs[6];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  ModulatedDeformConvBackwardCUDAKernelLauncher(
      input, weight, bias, ones, offset, mask, columns, grad_input, grad_weight,
      grad_bias, grad_offset, grad_mask, grad_output, kernel_h, kernel_w,
      stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
      deformable_group, with_bias, ctx, stream);
}

PARROTS_EXTENSION_REGISTER(modulated_deform_conv_forward)
    .attr("kernel_h")
    .attr("kernel_w")
    .attr("stride_h")
    .attr("stride_w")
    .attr("pad_h")
    .attr("pad_w")
    .attr("dilation_h")
    .attr("dilation_w")
    .attr("group")
    .attr("deformable_group")
    .attr("with_bias")
    .input(6)
    .output(2)
    .apply(modulated_deform_conv_forward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(modulated_deform_conv_backward)
    .attr("kernel_h")
    .attr("kernel_w")
    .attr("stride_h")
    .attr("stride_w")
    .attr("pad_h")
    .attr("pad_w")
    .attr("dilation_h")
    .attr("dilation_w")
    .attr("group")
    .attr("deformable_group")
    .attr("with_bias")
    .input(6)
    .output(7)
    .apply(modulated_deform_conv_backward_cuda)
    .done();
