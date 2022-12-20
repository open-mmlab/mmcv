#include <parrots/compute/aten.hpp>
#include <parrots/darray/darraymath.hpp>

#include "parrots_nms.h"

using namespace parrots;

using at::Tensor;
Tensor nms_cpu(Tensor boxes, Tensor scores, float iou_threshold, int offset);

void nms_parrots_cpu(HostContext &ctx, const SSElement &attr,
                     const OperatorBase::in_list_t &ins,
                     OperatorBase::out_list_t &outs) {
  float iou_threshold;
  int offset;
  SSAttrs(attr)
      .get("iou_threshold", iou_threshold)
      .get("offset", offset)
      .done();

  at::Tensor boxes, scores;
  boxes = buildATensor(ctx, ins[0]);
  scores = buildATensor(ctx, ins[1]);
  auto out = nms_cpu(boxes, scores, iou_threshold, offset);
  updateDArray(ctx, out, outs[0]);
  return;
}

REGISTER_DEVICE_IMPL(nms_impl, CPU, Arch::X86, nms_parrots_cpu);
