#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "deform_conv_pytorch.h"

using namespace parrots;

/*void deform_conv_forward_cuda(Tensor input, Tensor weight, Tensor offset,
 *                              Tensor output, Tensor columns, Tensor ones,
 *                              int kW, int kH, int dW, int dH, int padW,
 *                              int padH, int dilationW, int dilationH, int
 * group, int deformable_group, int im2col_step);
 */
void deform_conv_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
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

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& weight = buildATensor(ctx, ins[1]);
  const auto& offset = buildATensor(ctx, ins[2]);

  auto output = buildATensor(ctx, outs[0]);
  auto columns = buildATensor(ctx, outs[1]);
  auto ones = buildATensor(ctx, outs[2]);

  deform_conv_forward_cuda(input, weight, offset, output, columns, ones, kW, kH,
                           dW, dH, padW, padH, dilationW, dilationH, group,
                           deformable_group, im2col_step);
}

/*void deform_conv_backward_input_cuda(Tensor input, Tensor offset,
 *                                     Tensor gradOutput, Tensor gradInput,
 *                                     Tensor gradOffset, Tensor weight,
 *                                     Tensor columns, int kW, int kH, int dW,
 *                                     int dH, int padW, int padH, int
 * dilationW, int dilationH, int group, int deformable_group, int im2col_step);
 */
void deform_conv_backward_input_cuda_parrots(CudaContext& ctx,
                                             const SSElement& attr,
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

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& offset = buildATensor(ctx, ins[1]);
  const auto& gradOutput = buildATensor(ctx, ins[2]);

  auto gradInput = buildATensor(ctx, outs[0]);
  auto gradOffset = buildATensor(ctx, outs[1]);
  auto weight = buildATensor(ctx, outs[2]);
  auto columns = buildATensor(ctx, outs[3]);

  deform_conv_backward_input_cuda(input, offset, gradOutput, gradInput,
                                  gradOffset, weight, columns, kW, kH, dW, dH,
                                  padW, padH, dilationW, dilationH, group,
                                  deformable_group, im2col_step);
}

/*void deform_conv_backward_parameters_cuda(
 *     Tensor input, Tensor offset, Tensor gradOutput, Tensor gradWeight,
 *     Tensor columns, Tensor ones, int kW, int kH, int dW, int dH, int padW,
 *     int padH, int dilationW, int dilationH, int group, int deformable_group,
 *     float scale, int im2col_step);
 */
void deform_conv_backward_parameters_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
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

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& offset = buildATensor(ctx, ins[1]);
  const auto& gradOutput = buildATensor(ctx, ins[2]);

  auto gradWeight = buildATensor(ctx, outs[0]);
  auto columns = buildATensor(ctx, outs[1]);
  auto ones = buildATensor(ctx, outs[2]);
  deform_conv_backward_parameters_cuda(input, offset, gradOutput, gradWeight,
                                       columns, ones, kW, kH, dW, dH, padW,
                                       padH, dilationW, dilationH, group,
                                       deformable_group, scale, im2col_step);
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
    .apply(deform_conv_forward_cuda_parrots)
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
    .apply(deform_conv_backward_input_cuda_parrots)
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
    .apply(deform_conv_backward_parameters_cuda_parrots)
    .done();
