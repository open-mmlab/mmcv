#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "cc_attention_pytorch.h"

using namespace parrots;

/*void ca_forward_cuda(const Tensor t, const Tensor f, Tensor weight);*/
void ca_forward_cuda_parrots(CudaContext &ctx, const SSElement &attr,
                             const OperatorBase::in_list_t &ins,
                             OperatorBase::out_list_t &outs) {
  const auto &t = buildATensor(ctx, ins[0]);
  const auto &f = buildATensor(ctx, ins[1]);
  auto weight = buildATensor(ctx, outs[0]);
  ca_forward_cuda(t, f, weight);
}

/* void ca_backward_cuda(const Tensor dw, const Tensor t, const Tensor f,
 *                       Tensor dt, Tensor df)
 */
void ca_backward_cuda_parrots(CudaContext &ctx, const SSElement &attr,
                              const OperatorBase::in_list_t &ins,
                              OperatorBase::out_list_t &outs) {
  const auto &dw = buildATensor(ctx, ins[0]);
  const auto &t = buildATensor(ctx, ins[1]);
  const auto &f = buildATensor(ctx, ins[2]);
  auto dt = buildATensor(ctx, outs[0]);
  auto df = buildATensor(ctx, outs[1]);
  ca_backward_cuda(dw, t, f, dt, df);
}

/* void ca_map_forward_cuda(const Tensor weight, const Tensor g, Tensor out); */
void ca_map_forward_cuda_parrots(CudaContext &ctx, const SSElement &attr,
                                 const OperatorBase::in_list_t &ins,
                                 OperatorBase::out_list_t &outs) {
  const auto &weight = buildATensor(ctx, ins[0]);
  const auto &g = buildATensor(ctx, ins[1]);
  auto out = buildATensor(ctx, outs[0]);
  ca_map_forward_cuda(weight, g, out);
}

/* void ca_map_backward_cuda(const Tensor dout, const Tensor weight,
 *                           const Tensor g, Tensor dw, Tensor dg);
 */
void ca_map_backward_cuda_parrots(CudaContext &ctx, const SSElement &attr,
                                  const OperatorBase::in_list_t &ins,
                                  OperatorBase::out_list_t &outs) {
  const auto &dout = buildATensor(ctx, ins[0]);
  const auto &weight = buildATensor(ctx, ins[1]);
  const auto &g = buildATensor(ctx, ins[2]);
  auto dw = buildATensor(ctx, outs[0]);
  auto dg = buildATensor(ctx, outs[1]);
  ca_map_backward_cuda(dout, weight, g, dw, dg);
}

PARROTS_EXTENSION_REGISTER(ca_forward)
    .input(2)
    .output(1)
    .apply(ca_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(ca_backward)
    .input(3)
    .output(2)
    .apply(ca_backward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(ca_map_forward)
    .input(2)
    .output(1)
    .apply(ca_map_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(ca_map_backward)
    .input(3)
    .output(2)
    .apply(ca_map_backward_cuda_parrots)
    .done();
