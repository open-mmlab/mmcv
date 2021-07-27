#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "corner_pool_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void bottom_pool_forward_parrots(CudaContext& ctx, const SSElement& attr,
                                 const OperatorBase::in_list_t& ins,
                                 OperatorBase::out_list_t& outs) {
  at::Tensor input;
  input = buildATensor(ctx, ins[0]);
  auto out = bottom_pool_forward(input);
  updateDArray(ctx, out, outs[0]);
}

void bottom_pool_backward_parrots(CudaContext& ctx, const SSElement& attr,
                                  const OperatorBase::in_list_t& ins,
                                  OperatorBase::out_list_t& outs) {
  at::Tensor input, grad_output;
  input = buildATensor(ctx, ins[0]);
  grad_output = buildATensor(ctx, ins[1]);
  auto out = bottom_pool_backward(input, grad_output);
  updateDArray(ctx, out, outs[0]);
}

void left_pool_forward_parrots(CudaContext& ctx, const SSElement& attr,
                               const OperatorBase::in_list_t& ins,
                               OperatorBase::out_list_t& outs) {
  at::Tensor input;
  input = buildATensor(ctx, ins[0]);
  auto out = left_pool_forward(input);
  updateDArray(ctx, out, outs[0]);
}

void left_pool_backward_parrots(CudaContext& ctx, const SSElement& attr,
                                const OperatorBase::in_list_t& ins,
                                OperatorBase::out_list_t& outs) {
  at::Tensor input, grad_output;
  input = buildATensor(ctx, ins[0]);
  grad_output = buildATensor(ctx, ins[1]);
  auto out = left_pool_backward(input, grad_output);
  updateDArray(ctx, out, outs[0]);
}

void right_pool_forward_parrots(CudaContext& ctx, const SSElement& attr,
                                const OperatorBase::in_list_t& ins,
                                OperatorBase::out_list_t& outs) {
  at::Tensor input;
  input = buildATensor(ctx, ins[0]);
  auto out = right_pool_forward(input);
  updateDArray(ctx, out, outs[0]);
}

void right_pool_backward_parrots(CudaContext& ctx, const SSElement& attr,
                                 const OperatorBase::in_list_t& ins,
                                 OperatorBase::out_list_t& outs) {
  at::Tensor input, grad_output;
  input = buildATensor(ctx, ins[0]);
  grad_output = buildATensor(ctx, ins[1]);
  auto out = right_pool_backward(input, grad_output);
  updateDArray(ctx, out, outs[0]);
}

void top_pool_forward_parrots(CudaContext& ctx, const SSElement& attr,
                              const OperatorBase::in_list_t& ins,
                              OperatorBase::out_list_t& outs) {
  at::Tensor input;
  input = buildATensor(ctx, ins[0]);
  auto out = top_pool_forward(input);
  updateDArray(ctx, out, outs[0]);
}

void top_pool_backward_parrots(CudaContext& ctx, const SSElement& attr,
                               const OperatorBase::in_list_t& ins,
                               OperatorBase::out_list_t& outs) {
  at::Tensor input, grad_output;
  input = buildATensor(ctx, ins[0]);
  grad_output = buildATensor(ctx, ins[1]);
  auto out = top_pool_backward(input, grad_output);
  updateDArray(ctx, out, outs[0]);
}
#endif

void bottom_pool_forward_parrots_cpu(HostContext& ctx, const SSElement& attr,
                                     const OperatorBase::in_list_t& ins,
                                     OperatorBase::out_list_t& outs) {
  at::Tensor input;
  input = buildATensor(ctx, ins[0]);
  auto out = bottom_pool_forward(input);
  updateDArray(ctx, out, outs[0]);
}

void bottom_pool_backward_parrots_cpu(HostContext& ctx, const SSElement& attr,
                                      const OperatorBase::in_list_t& ins,
                                      OperatorBase::out_list_t& outs) {
  at::Tensor input, grad_output;
  input = buildATensor(ctx, ins[0]);
  grad_output = buildATensor(ctx, ins[1]);
  auto out = bottom_pool_backward(input, grad_output);
  updateDArray(ctx, out, outs[0]);
}

void left_pool_forward_parrots_cpu(HostContext& ctx, const SSElement& attr,
                                   const OperatorBase::in_list_t& ins,
                                   OperatorBase::out_list_t& outs) {
  at::Tensor input;
  input = buildATensor(ctx, ins[0]);
  auto out = left_pool_forward(input);
  updateDArray(ctx, out, outs[0]);
}

void left_pool_backward_parrots_cpu(HostContext& ctx, const SSElement& attr,
                                    const OperatorBase::in_list_t& ins,
                                    OperatorBase::out_list_t& outs) {
  at::Tensor input, grad_output;
  input = buildATensor(ctx, ins[0]);
  grad_output = buildATensor(ctx, ins[1]);
  auto out = left_pool_backward(input, grad_output);
  updateDArray(ctx, out, outs[0]);
}

void right_pool_forward_parrots_cpu(HostContext& ctx, const SSElement& attr,
                                    const OperatorBase::in_list_t& ins,
                                    OperatorBase::out_list_t& outs) {
  at::Tensor input;
  input = buildATensor(ctx, ins[0]);
  auto out = right_pool_forward(input);
  updateDArray(ctx, out, outs[0]);
}

void right_pool_backward_parrots_cpu(HostContext& ctx, const SSElement& attr,
                                     const OperatorBase::in_list_t& ins,
                                     OperatorBase::out_list_t& outs) {
  at::Tensor input, grad_output;
  input = buildATensor(ctx, ins[0]);
  grad_output = buildATensor(ctx, ins[1]);
  auto out = right_pool_backward(input, grad_output);
  updateDArray(ctx, out, outs[0]);
}

void top_pool_forward_parrots_cpu(HostContext& ctx, const SSElement& attr,
                                  const OperatorBase::in_list_t& ins,
                                  OperatorBase::out_list_t& outs) {
  at::Tensor input;
  input = buildATensor(ctx, ins[0]);
  auto out = top_pool_forward(input);
  updateDArray(ctx, out, outs[0]);
}

void top_pool_backward_parrots_cpu(HostContext& ctx, const SSElement& attr,
                                   const OperatorBase::in_list_t& ins,
                                   OperatorBase::out_list_t& outs) {
  at::Tensor input, grad_output;
  input = buildATensor(ctx, ins[0]);
  grad_output = buildATensor(ctx, ins[1]);
  auto out = top_pool_backward(input, grad_output);
  updateDArray(ctx, out, outs[0]);
}

PARROTS_EXTENSION_REGISTER(bottom_pool_forward)
    .input(1)
    .output(1)
#ifdef MMCV_WITH_CUDA
    .apply(bottom_pool_forward_parrots)
#endif
    .apply(bottom_pool_forward_parrots_cpu)
    .done();

PARROTS_EXTENSION_REGISTER(bottom_pool_backward)
    .input(2)
    .output(1)
#ifdef MMCV_WITH_CUDA
    .apply(bottom_pool_backward_parrots)
#endif
    .apply(bottom_pool_backward_parrots_cpu)
    .done();

PARROTS_EXTENSION_REGISTER(top_pool_forward)
    .input(1)
    .output(1)
#ifdef MMCV_WITH_CUDA
    .apply(top_pool_forward_parrots)
#endif
    .apply(top_pool_forward_parrots_cpu)
    .done();

PARROTS_EXTENSION_REGISTER(top_pool_backward)
    .input(2)
    .output(1)
#ifdef MMCV_WITH_CUDA
    .apply(top_pool_backward_parrots)
#endif
    .apply(top_pool_backward_parrots_cpu)
    .done();

PARROTS_EXTENSION_REGISTER(left_pool_forward)
    .input(1)
    .output(1)
#ifdef MMCV_WITH_CUDA
    .apply(left_pool_forward_parrots)
#endif
    .apply(left_pool_forward_parrots_cpu)
    .done();

PARROTS_EXTENSION_REGISTER(left_pool_backward)
    .input(2)
    .output(1)
#ifdef MMCV_WITH_CUDA
    .apply(left_pool_backward_parrots)
#endif
    .apply(left_pool_backward_parrots_cpu)
    .done();

PARROTS_EXTENSION_REGISTER(right_pool_forward)
    .input(1)
    .output(1)
#ifdef MMCV_WITH_CUDA
    .apply(right_pool_forward_parrots)
#endif
    .apply(right_pool_forward_parrots_cpu)
    .done();

PARROTS_EXTENSION_REGISTER(right_pool_backward)
    .input(2)
    .output(1)
#ifdef MMCV_WITH_CUDA
    .apply(right_pool_backward_parrots)
#endif
    .apply(right_pool_backward_parrots_cpu)
    .done();
