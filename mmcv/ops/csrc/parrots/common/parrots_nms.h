#ifndef PARROTS_NMS_H
#define PARROTS_NMS_H

#include <parrots_device_registry.hpp>

using namespace parrots;

void nms_impl(Context &ctx, const SSElement &attr,
              const OperatorBase::in_list_t &ins,
              OperatorBase::out_list_t &outs);

#endif  // PARROTS_NMS_H
