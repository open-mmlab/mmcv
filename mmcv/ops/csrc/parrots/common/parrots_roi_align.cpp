
#include "parrots_roi_align.h"

using namespace parrots;

void roi_align_forward_impl(Context &ctx, const SSElement &attr,
                            const OperatorBase::in_list_t &ins,
                            OperatorBase::out_list_t &outs) {
  return DISPATCH_DEVICE_IMPL(roi_align_forward_impl, ctx, attr, ins, outs);
}

void roi_align_backward_impl(Context &ctx, const SSElement &attr,
                             const OperatorBase::in_list_t &ins,
                             OperatorBase::out_list_t &outs) {
  return DISPATCH_DEVICE_IMPL(roi_align_backward_impl, ctx, attr, ins, outs);
}

PARROTS_EXTENSION_REGISTER(roi_align_forward)
    .attr("aligned_height")
    .attr("aligned_width")
    .attr("spatial_scale")
    .attr("sampling_ratio")
    .attr("pool_mode")
    .attr("aligned")
    .input(2)
    .output(3)
    .apply(roi_align_forward_impl)
    .done();

PARROTS_EXTENSION_REGISTER(roi_align_backward)
    .attr("aligned_height")
    .attr("aligned_width")
    .attr("spatial_scale")
    .attr("sampling_ratio")
    .attr("pool_mode")
    .attr("aligned")
    .input(4)
    .output(1)
    .apply(roi_align_backward_impl)
    .done();
