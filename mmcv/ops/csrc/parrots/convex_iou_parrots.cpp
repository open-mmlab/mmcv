// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "convex_iou_pytorch.h"
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <parrots/diopi.hpp>

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void convex_iou_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                     const OperatorBase::in_list_t& ins,
                                     OperatorBase::out_list_t& outs) {
  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto pointsets = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));
  auto polygons = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[1]));
  auto ious = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);
  PARROTS_CALLDIOPI(diopiConvexIou(ch, pointsets, polygons, ious));
}

void convex_giou_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                      const OperatorBase::in_list_t& ins,
                                      OperatorBase::out_list_t& outs) {
  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto pointsets = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));
  auto polygons = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[1]));
  auto output = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);
  PARROTS_CALLDIOPI(diopiConvexGiou(ch, pointsets, polygons, output));
}

PARROTS_EXTENSION_REGISTER(convex_iou)
    .input(2)
    .output(1)
    .apply(convex_iou_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(convex_giou)
    .input(2)
    .output(1)
    .apply(convex_giou_forward_cuda_parrots)
    .done();

#endif
