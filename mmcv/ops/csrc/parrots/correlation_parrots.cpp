// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "correlation_pytorch.h"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include <parrots/diopi.hpp>
#endif

using namespace parrots;

#ifdef MMCV_WITH_CUDA
#ifdef MMCV_WITH_DIOPI
void correlation_forward_cuda_parrots_diopi(CudaContext& ctx,
                                            const SSElement& attr,
                                            const OperatorBase::in_list_t& ins,
                                            OperatorBase::out_list_t& outs) {
  int kH, kW, patchH, patchW, padH, padW, dilationH, dilationW, dilation_patchH,
      dilation_patchW, dH, dW;
  SSAttrs(attr)
      .get<int>("kH", kH)
      .get<int>("kW", kW)
      .get<int>("patchH", patchH)
      .get<int>("patchW", patchW)
      .get<int>("padH", padH)
      .get<int>("padW", padW)
      .get<int>("dilationH", dilationH)
      .get<int>("dilationW", dilationW)
      .get<int>("dilation_patchH", dilation_patchH)
      .get<int>("dilation_patchW", dilation_patchW)
      .get<int>("dH", dH)
      .get<int>("dW", dW)
      .done();

  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto input1 =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));
  auto input2 =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[1]));

  auto output = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);

  PARROTS_CALLDIOPI(diopiCorrelation(ch, input1, input2, output, kH, kW, patchH,
                                     patchW, padH, padW, dilationH, dilationW,
                                     dilation_patchH, dilation_patchW, dH, dW));
}

void correlation_backward_cuda_parrots_diopi(CudaContext& ctx,
                                             const SSElement& attr,
                                             const OperatorBase::in_list_t& ins,
                                             OperatorBase::out_list_t& outs) {
  int kH, kW, patchH, patchW, padH, padW, dilationH, dilationW, dilation_patchH,
      dilation_patchW, dH, dW;
  SSAttrs(attr)
      .get<int>("kH", kH)
      .get<int>("kW", kW)
      .get<int>("patchH", patchH)
      .get<int>("patchW", patchW)
      .get<int>("padH", padH)
      .get<int>("padW", padW)
      .get<int>("dilationH", dilationH)
      .get<int>("dilationW", dilationW)
      .get<int>("dilation_patchH", dilation_patchH)
      .get<int>("dilation_patchW", dilation_patchW)
      .get<int>("dH", dH)
      .get<int>("dW", dW)
      .done();

  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto grad_output =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));
  auto input1 =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[1]));
  auto input2 =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[2]));

  auto grad_input1 = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);
  auto grad_input2 = reinterpret_cast<diopiTensorHandle_t>(&outs[1]);

  PARROTS_CALLDIOPI(diopiCorrelationBackward(
      ch, grad_output, input1, input2, grad_input1, grad_input2, kH, kW, patchH,
      patchW, padH, padW, dilationH, dilationW, dilation_patchH,
      dilation_patchW, dH, dW));
}
#else
void correlation_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                      const OperatorBase::in_list_t& ins,
                                      OperatorBase::out_list_t& outs) {
  int kH, kW, patchH, patchW, padH, padW, dilationH, dilationW, dilation_patchH,
      dilation_patchW, dH, dW;
  SSAttrs(attr)
      .get<int>("kH", kH)
      .get<int>("kW", kW)
      .get<int>("patchH", patchH)
      .get<int>("patchW", patchW)
      .get<int>("padH", padH)
      .get<int>("padW", padW)
      .get<int>("dilationH", dilationH)
      .get<int>("dilationW", dilationW)
      .get<int>("dilation_patchH", dilation_patchH)
      .get<int>("dilation_patchW", dilation_patchW)
      .get<int>("dH", dH)
      .get<int>("dW", dW)
      .done();

  auto input1 = buildATensor(ctx, ins[0]);
  auto input2 = buildATensor(ctx, ins[1]);

  auto output = buildATensor(ctx, outs[0]);

  correlation_forward(input1, input2, output, kH, kW, patchH, patchW, padH,
                      padW, dilationH, dilationW, dilation_patchH,
                      dilation_patchW, dH, dW);
}

void correlation_backward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                       const OperatorBase::in_list_t& ins,
                                       OperatorBase::out_list_t& outs) {
  int kH, kW, patchH, patchW, padH, padW, dilationH, dilationW, dilation_patchH,
      dilation_patchW, dH, dW;
  SSAttrs(attr)
      .get<int>("kH", kH)
      .get<int>("kW", kW)
      .get<int>("patchH", patchH)
      .get<int>("patchW", patchW)
      .get<int>("padH", padH)
      .get<int>("padW", padW)
      .get<int>("dilationH", dilationH)
      .get<int>("dilationW", dilationW)
      .get<int>("dilation_patchH", dilation_patchH)
      .get<int>("dilation_patchW", dilation_patchW)
      .get<int>("dH", dH)
      .get<int>("dW", dW)
      .done();

  auto grad_output = buildATensor(ctx, ins[0]);
  auto input1 = buildATensor(ctx, ins[1]);
  auto input2 = buildATensor(ctx, ins[2]);

  auto grad_input1 = buildATensor(ctx, outs[0]);
  auto grad_input2 = buildATensor(ctx, outs[1]);

  correlation_backward(grad_output, input1, input2, grad_input1, grad_input2,
                       kH, kW, patchH, patchW, padH, padW, dilationH, dilationW,
                       dilation_patchH, dilation_patchW, dH, dW);
}
#endif
#endif

void correlation_forward_cpu_parrots(HostContext& ctx, const SSElement& attr,
                                     const OperatorBase::in_list_t& ins,
                                     OperatorBase::out_list_t& outs) {
  int kH, kW, patchH, patchW, padH, padW, dilationH, dilationW, dilation_patchH,
      dilation_patchW, dH, dW;
  SSAttrs(attr)
      .get<int>("kH", kH)
      .get<int>("kW", kW)
      .get<int>("patchH", patchH)
      .get<int>("patchW", patchW)
      .get<int>("padH", padH)
      .get<int>("padW", padW)
      .get<int>("dilationH", dilationH)
      .get<int>("dilationW", dilationW)
      .get<int>("dilation_patchH", dilation_patchH)
      .get<int>("dilation_patchW", dilation_patchW)
      .get<int>("dH", dH)
      .get<int>("dW", dW)
      .done();

  auto input1 = buildATensor(ctx, ins[0]);
  auto input2 = buildATensor(ctx, ins[1]);

  auto output = buildATensor(ctx, outs[0]);

  correlation_forward(input1, input2, output, kH, kW, patchH, patchW, padH,
                      padW, dilationH, dilationW, dilation_patchH,
                      dilation_patchW, dH, dW);
}

void correlation_backward_cpu_parrots(HostContext& ctx, const SSElement& attr,
                                      const OperatorBase::in_list_t& ins,
                                      OperatorBase::out_list_t& outs) {
  int kH, kW, patchH, patchW, padH, padW, dilationH, dilationW, dilation_patchH,
      dilation_patchW, dH, dW;
  SSAttrs(attr)
      .get<int>("kH", kH)
      .get<int>("kW", kW)
      .get<int>("patchH", patchH)
      .get<int>("patchW", patchW)
      .get<int>("padH", padH)
      .get<int>("padW", padW)
      .get<int>("dilationH", dilationH)
      .get<int>("dilationW", dilationW)
      .get<int>("dilation_patchH", dilation_patchH)
      .get<int>("dilation_patchW", dilation_patchW)
      .get<int>("dH", dH)
      .get<int>("dW", dW)
      .done();

  auto grad_output = buildATensor(ctx, ins[0]);
  auto input1 = buildATensor(ctx, ins[1]);
  auto input2 = buildATensor(ctx, ins[2]);

  auto grad_input1 = buildATensor(ctx, outs[0]);
  auto grad_input2 = buildATensor(ctx, outs[1]);

  correlation_backward(grad_output, input1, input2, grad_input1, grad_input2,
                       kH, kW, patchH, patchW, padH, padW, dilationH, dilationW,
                       dilation_patchH, dilation_patchW, dH, dW);
}

PARROTS_EXTENSION_REGISTER(correlation_forward)
    .attr("kH")
    .attr("kW")
    .attr("patchH")
    .attr("patchW")
    .attr("padH")
    .attr("padW")
    .attr("dilationH")
    .attr("dilationW")
    .attr("dilation_patchH")
    .attr("dilation_patchW")
    .attr("dH")
    .attr("dW")
    .input(2)
    .output(1)
    .apply(correlation_forward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
#ifdef MMCV_WITH_DIOPI
    .apply(correlation_forward_cuda_parrots_diopi)
#else
    .apply(correlation_forward_cuda_parrots)
#endif
#endif
    .done();

PARROTS_EXTENSION_REGISTER(correlation_backward)
    .attr("kH")
    .attr("kW")
    .attr("patchH")
    .attr("patchW")
    .attr("padH")
    .attr("padW")
    .attr("dilationH")
    .attr("dilationW")
    .attr("dilation_patchH")
    .attr("dilation_patchW")
    .attr("dH")
    .attr("dW")
    .input(3)
    .output(2)
    .apply(correlation_backward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
#ifdef MMCV_WITH_DIOPI
    .apply(correlation_backward_cuda_parrots_diopi)
#else
    .apply(correlation_backward_cuda_parrots)
#endif
#endif
    .done();
