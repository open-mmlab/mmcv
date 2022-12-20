
#include "parrots_nms.h"

using namespace parrots;

void nms_impl(Context &ctx, const SSElement &attr,
              const OperatorBase::in_list_t &ins,
              OperatorBase::out_list_t &outs) {
  return DISPATCH_DEVICE_IMPL(nms_impl, ctx, attr, ins, outs);
}

PARROTS_EXTENSION_REGISTER(nms)
    .attr("iou_threshold")
    .attr("offset")
    .input(2)
    .output(1)
    .apply(nms_impl)
    .done();
