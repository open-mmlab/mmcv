#ifndef PARROTS_FOCAL_LOSS_H
#define PARROTS_FOCAL_LOSS_H

#include <parrots_device_registry.hpp>

using namespace parrots;

void sigmoid_focal_loss_forward_impl(Context &ctx, const SSElement &attr,
                                     const OperatorBase::in_list_t &ins,
                                     OperatorBase::out_list_t &outs);

void sigmoid_focal_loss_backward_impl(Context &ctx, const SSElement &attr,
                                      const OperatorBase::in_list_t &ins,
                                      OperatorBase::out_list_t &outs);

#endif  // PARROTS_FOCAL_LOSS_H
