
#include "parrots_focal_loss.h"

using namespace parrots;

void sigmoid_focal_loss_forward_impl(Context &ctx, const SSElement &attr,
                                     const OperatorBase::in_list_t &ins,
                                     OperatorBase::out_list_t &outs) {
  return DISPATCH_DEVICE_IMPL(sigmoid_focal_loss_forward_impl, ctx, attr, ins,
                              outs);
}

void sigmoid_focal_loss_backward_impl(Context &ctx, const SSElement &attr,
                                      const OperatorBase::in_list_t &ins,
                                      OperatorBase::out_list_t &outs) {
  return DISPATCH_DEVICE_IMPL(sigmoid_focal_loss_backward_impl, ctx, attr, ins,
                              outs);
}

PARROTS_EXTENSION_REGISTER(sigmoid_focal_loss_forward)
    .attr("gamma")
    .attr("alpha")
    .input(3)
    .output(1)
    .apply(sigmoid_focal_loss_forward_impl)
    .done();

PARROTS_EXTENSION_REGISTER(sigmoid_focal_loss_backward)
    .attr("gamma")
    .attr("alpha")
    .input(3)
    .output(1)
    .apply(sigmoid_focal_loss_backward_impl)
    .done();
