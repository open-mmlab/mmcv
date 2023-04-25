// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "convex_iou_pytorch.h"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include <parrots/diopi.hpp>
#endif

using namespace parrots;

#ifdef MMCV_WITH_CUDA
#ifdef MMCV_WITH_DIOPI
void convex_iou_forward_cuda_parrots_diopi(CudaContext& ctx,
                                           const SSElement& attr,
                                           const OperatorBase::in_list_t& ins,
                                           OperatorBase::out_list_t& outs) {
  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto pointsets =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));
  auto polygons =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[1]));
  auto ious = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);
  PARROTS_CALLDIOPI(diopiConvexIou(ch, pointsets, polygons, ious));
}

void convex_giou_forward_cuda_parrots_diopi(CudaContext& ctx,
                                            const SSElement& attr,
                                            const OperatorBase::in_list_t& ins,
                                            OperatorBase::out_list_t& outs) {
  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto pointsets =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));
  auto polygons =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[1]));
  auto output = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);
  PARROTS_CALLDIOPI(diopiConvexGiou(ch, pointsets, polygons, output));
}
#else
void convex_iou_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                     const OperatorBase::in_list_t& ins,
                                     OperatorBase::out_list_t& outs) {
  auto pointsets = buildATensor(ctx, ins[0]);
  auto polygons = buildATensor(ctx, ins[1]);
  auto ious = buildATensor(ctx, outs[0]);
  convex_iou(pointsets, polygons, ious);
}

void convex_giou_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                      const OperatorBase::in_list_t& ins,
                                      OperatorBase::out_list_t& outs) {
  auto pointsets = buildATensor(ctx, ins[0]);
  auto polygons = buildATensor(ctx, ins[1]);
  auto output = buildATensor(ctx, outs[0]);
  convex_giou(pointsets, polygons, output);
}
#endif

PARROTS_EXTENSION_REGISTER(convex_iou)
    .input(2)
    .output(1)
#ifdef MMCV_WITH_DIOPI
    .apply(convex_iou_forward_cuda_parrots_diopi)
#else
    .apply(convex_iou_forward_cuda_parrots)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(convex_giou)
    .input(2)
    .output(1)
#ifdef MMCV_WITH_DIOPI
    .apply(convex_giou_forward_cuda_parrots_diopi)
#else
    .apply(convex_giou_forward_cuda_parrots)
#endif
    .done();
#endif
