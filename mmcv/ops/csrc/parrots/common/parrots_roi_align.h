#ifndef PARROTS_ROI_ALIGN_H
#define PARROTS_ROI_ALIGN_H

#include <parrots_device_registry.hpp>

using namespace parrots;

void roi_align_forward_impl(Context &ctx, const SSElement &attr,
                            const OperatorBase::in_list_t &ins,
                            OperatorBase::out_list_t &outs);

void roi_align_backward_impl(Context &ctx, const SSElement &attr,
                             const OperatorBase::in_list_t &ins,
                             OperatorBase::out_list_t &outs);

#endif  // PARROTS_ROI_ALIGN_H
