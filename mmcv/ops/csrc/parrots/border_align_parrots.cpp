// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "border_align_pytorch.h"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include <parrots/diopi.hpp>
#endif

using namespace parrots;

#ifdef MMCV_WITH_CUDA
#ifdef MMCV_WITH_DIOPI
void border_align_forward_cuda_parrots_diopi(CudaContext& ctx,
                                             const SSElement& attr,
                                             const OperatorBase::in_list_t& ins,
                                             OperatorBase::out_list_t& outs) {
  int pool_size;
  SSAttrs(attr).get<int>("pool_size", pool_size).done();

  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto input =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));
  auto boxes =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[1]));

  auto output = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);
  auto argmax_idx = reinterpret_cast<diopiTensorHandle_t>(&outs[1]);
  PARROTS_CALLDIOPI(
      diopiBorderAlignMmcv(ch, output, argmax_idx, input, boxes, pool_size));
}

void border_align_backward_cuda_parrots_diopi(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  int pool_size;
  SSAttrs(attr).get<int>("pool_size", pool_size).done();

  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto top_grad =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));
  auto boxes =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[1]));
  auto argmax_idx =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[2]));

  auto bottom_grad = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);
  PARROTS_CALLDIOPI(diopiBorderAlignBackwardMmcv(ch, top_grad, boxes, argmax_idx,
                                             bottom_grad, pool_size));
}
#else
void border_align_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                       const OperatorBase::in_list_t& ins,
                                       OperatorBase::out_list_t& outs) {
  int pool_size;
  SSAttrs(attr).get<int>("pool_size", pool_size).done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& boxes = buildATensor(ctx, ins[1]);

  auto output = buildATensor(ctx, outs[0]);
  auto argmax_idx = buildATensor(ctx, outs[1]);
  border_align_forward_cuda(input, boxes, output, argmax_idx, pool_size);
}

void border_align_backward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                        const OperatorBase::in_list_t& ins,
                                        OperatorBase::out_list_t& outs) {
  int pool_size;
  SSAttrs(attr).get<int>("pool_size", pool_size).done();

  const auto& top_grad = buildATensor(ctx, ins[0]);
  const auto& boxes = buildATensor(ctx, ins[1]);
  const auto& argmax_idx = buildATensor(ctx, ins[2]);

  auto bottom_grad = buildATensor(ctx, outs[0]);
  border_align_backward_cuda(top_grad, boxes, argmax_idx, bottom_grad,
                             pool_size);
}
#endif

PARROTS_EXTENSION_REGISTER(border_align_forward)
    .attr("pool_size")
    .input(2)
    .output(2)
#ifdef MMCV_WITH_DIOPI
    .apply(border_align_forward_cuda_parrots_diopi)
#else
    .apply(border_align_forward_cuda_parrots)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(border_align_backward)
    .attr("pool_size")
    .input(3)
    .output(1)
#ifdef MMCV_WITH_DIOPI
    .apply(border_align_forward_cuda_parrots_diopi)
#else
    .apply(border_align_backward_cuda_parrots)
#endif
    .done();
#endif
