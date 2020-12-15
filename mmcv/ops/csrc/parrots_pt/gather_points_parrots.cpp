#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "gather_points_pytorch.h"

using namespace parrots;

void gather_points_parrots(CudaContext& ctx, const SSElement& attr,
                           const OperatorBase::in_list_t& ins,
                           OperatorBase::out_list_t& outs) {
  int b, c, n, npoints;
  SSAttrs(attr)
      .get("b", b)
      .get("c", c)
      .get("n", n)
      .get("npoints", npoints)
      .done();

  at::Tensor points_tensor, idx_tensor, out_tensor;
  points_tensor = buildATensor(ctx, ins[0]);
  idx_tensor = buildATensor(ctx, ins[1]);
  out_tensor = buildATensor(ctx, outs[0]);
  gather_points(b, c, n, npoints, points_tensor, idx_tensor, out_tensor);
}

void gather_points_backward_parrots(CudaContext& ctx, const SSElement& attr,
                                    const OperatorBase::in_list_t& ins,
                                    OperatorBase::out_list_t& outs) {
  int b, c, n, npoints;
  SSAttrs(attr)
      .get("b", b)
      .get("c", c)
      .get("n", n)
      .get("npoints", npoints)
      .done();

  at::Tensor grad_out_tensor, idx_tensor, grad_points_tensor;
  grad_out_tensor = buildATensor(ctx, ins[0]);
  idx_tensor = buildATensor(ctx, ins[1]);
  grad_points_tensor = buildATensor(ctx, outs[0]);
  gather_points_backward(b, c, n, npoints, grad_out_tensor, idx_tensor,
                         grad_points_tensor);
}

PARROTS_EXTENSION_REGISTER(gather_points)
    .attr("b")
    .attr("c")
    .attr("n")
    .attr("npoints")
    .input(2)
    .output(1)
    .apply(gather_points_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(gather_points_backward)
    .attr("b")
    .attr("c")
    .attr("n")
    .attr("npoints")
    .input(2)
    .output(1)
    .apply(gather_points_backward_parrots)
    .done();
