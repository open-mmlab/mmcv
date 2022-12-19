// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "active_rotated_filter_pytorch.h"
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <parrots/diopi.hpp>

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void active_rotated_filter_forward_cuda_parrots_diopi(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto input = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));
  auto indices = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[1]));
  auto output = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);
  PARROTS_CALLDIOPI(diopiActiveRotatedFilter(ch, input, indices, output));
}

void active_rotated_filter_backward_cuda_parrots_diopi(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto grad_out = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));
  auto indices = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[1]));
  auto grad_in = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);
  PARROTS_CALLDIOPI(diopiActiveRotatedFilterBackward(ch, grad_out, indices, grad_in));
}
#endif

void active_rotated_filter_forward_cpu_parrots(
    HostContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  auto input = buildATensor(ctx, ins[0]);
  auto indices = buildATensor(ctx, ins[1]);
  auto output = buildATensor(ctx, outs[0]);
  active_rotated_filter_forward(input, indices, output);
}

void active_rotated_filter_backward_cpu_parrots(
    HostContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  auto grad_out = buildATensor(ctx, ins[0]);
  auto indices = buildATensor(ctx, ins[1]);
  auto grad_in = buildATensor(ctx, outs[0]);
  active_rotated_filter_backward(grad_out, indices, grad_in);
}

PARROTS_EXTENSION_REGISTER(active_rotated_filter_forward)
    .input(2)
    .output(1)
    .apply(active_rotated_filter_forward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(active_rotated_filter_forward_cuda_parrots_diopi)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(active_rotated_filter_backward)
    .input(2)
    .output(1)
    .apply(active_rotated_filter_backward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(active_rotated_filter_backward_cuda_parrots_diopi)
#endif
    .done();
