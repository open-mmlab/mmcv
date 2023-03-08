// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "diff_iou_rotated_pytorch.h"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <parrots/diopi.hpp>
#endif

using namespace parrots;

#ifdef MMCV_WITH_CUDA
#ifdef MMCV_WITH_DIOPI
void diff_iou_rotated_sort_vertices_forward_cuda_parrots_diopi(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto vertices = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));
  auto mask = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[1]));
  auto num_valid = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[2]));
  auto out = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);
  auto outhandle = &out;
  PARROTS_CALLDIOPI(diopiDiffIouRotatedSortVertices(ch, outhandle, vertices, mask, num_valid));
}
#else
void diff_iou_rotated_sort_vertices_forward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  at::Tensor boxes, scores, dets;
  auto vertices = buildATensor(ctx, ins[0]);
  auto mask = buildATensor(ctx, ins[1]);
  auto num_valid = buildATensor(ctx, ins[2]);
  auto out =
      diff_iou_rotated_sort_vertices_forward_cuda(vertices, mask, num_valid);
  updateDArray(ctx, out, outs[0]);
}
#endif

PARROTS_EXTENSION_REGISTER(diff_iou_rotated_sort_vertices_forward)
    .input(3)
    .output(1)
#ifdef MMCV_WITH_DIOPI
    .apply(diff_iou_rotated_sort_vertices_forward_cuda_parrots_diopi)
#else
    .apply(diff_iou_rotated_sort_vertices_forward_cuda_parrots)
#endif
    .done();
#endif
