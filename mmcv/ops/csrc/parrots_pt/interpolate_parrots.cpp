#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>
#include <parrots/compute/aten.hpp>
#include "interpolate_pytorch.h"
using namespace parrots;

void three_nn_parrots(CudaContext& ctx, const SSElement& attr,
                        const OperatorBase::in_list_t& ins,
                        OperatorBase::out_list_t& outs){
  int b, n, m;
  SSAttrs(attr)
    .get("b", b)
    .get("n", n)
    .get("m", m)
    .done();

  at::Tensor unknown_tensor, known_tensor, dist2_tensor, idx_tensor;
  unknown_tensor = buildATensor(ctx, ins[0]);
  known_tensor = buildATensor(ctx, ins[1]);
  dist2_tensor = buildATensor(ctx, outs[0]);
  idx_tensor = buildATensor(ctx, outs[1]);

  three_nn(b, n, m, unknown_tensor, known_tensor, dist2_tensor, idx_tensor);
}

void three_interpolate_parrots(CudaContext& ctx, const SSElement& attr,
                        const OperatorBase::in_list_t& ins,
                        OperatorBase::out_list_t& outs){
  int b, c, m, n;
  SSAttrs(attr)
    .get("b", b)
    .get("c", c)
    .get("m", m)
    .get("n", n)
    .done();

  at::Tensor points_tensor, idx_tensor, weight_tensor, out_tensor;
  points_tensor = buildATensor(ctx, ins[0]);
  idx_tensor = buildATensor(ctx, ins[1]);
  weight_tensor = buildATensor(ctx, ins[2]);
  out_tensor = buildATensor(ctx, outs[0]);

  three_interpolate(b, c, m, n, points_tensor, idx_tensor, weight_tensor, out_tensor);
}

void three_interpolate_backward_parrots(CudaContext& ctx, const SSElement& attr,
                        const OperatorBase::in_list_t& ins,
                        OperatorBase::out_list_t& outs){
  int b, c, n, m;
  SSAttrs(attr)
    .get("b", b)
    .get("c", c)
    .get("n", n)
    .get("m", m)
    .done();

  at::Tensor grad_out_tensor, idx_tensor, weight_tensor, grad_points_tensor;
  grad_out_tensor = buildATensor(ctx, ins[0]);
  idx_tensor = buildATensor(ctx, ins[1]);
  weight_tensor = buildATensor(ctx, ins[2]);
  grad_points_tensor = buildATensor(ctx, outs[0]);

  three_interpolate_backward(b, c, n, m, grad_out_tensor, idx_tensor, weight_tensor, grad_points_tensor);
}

PARROTS_EXTENSION_REGISTER(three_nn)
    .attr("b")
    .attr("n")
    .attr("m")
    .input(2)
    .output(2)
    .apply(three_nn_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(three_interpolate)
    .attr("b")
    .attr("c")
    .attr("m")
    .attr("n")
    .input(3)
    .output(1)
    .apply(three_interpolate_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(three_interpolate_backward)
    .attr("b")
    .attr("c")
    .attr("n")
    .attr("m")
    .input(3)
    .output(1)
    .apply(three_interpolate_backward_parrots)
    .done();
