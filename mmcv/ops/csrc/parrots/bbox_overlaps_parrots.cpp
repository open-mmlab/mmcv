// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "bbox_overlaps_pytorch.h"
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <parrots/diopi.hpp>

using namespace parrots;

#ifdef MMCV_WITH_CUDA
/*
 * void bbox_overlaps_cuda(const Tensor bboxes1, const Tensor bboxes2, Tensor
 * ious, const int mode, const bool aligned, const int offset);
 */
void bbox_overlaps_parrots_diopi(CudaContext& ctx, const SSElement& attr,
                           const OperatorBase::in_list_t& ins,
                           OperatorBase::out_list_t& outs) {
  int mode, offset;
  bool aligned;
  SSAttrs(attr)
      .get<int>("mode", mode)
      .get<bool>("aligned", aligned)
      .get<int>("offset", offset)
      .done();

  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto bboxes1 = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));
  auto bboxes2 = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[1]));
  auto ious = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);
  PARROTS_CALLDIOPI(diopiBboxOverlaps(ch, bboxes1, bboxes2, ious, mode, aligned, offset));
}

PARROTS_EXTENSION_REGISTER(bbox_overlaps)
    .attr("mode")
    .attr("aligned")
    .attr("offset")
    .input(2)
    .output(1)
    .apply(bbox_overlaps_parrots_diopi)
    .done();
#endif
