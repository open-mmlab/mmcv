#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "group_points_pytorch.h"

using namespace parrots;

void group_points_parrots(CudaContext& ctx, const SSElement& attr,
                          const OperatorBase::in_list_t& ins,
                          OperatorBase::out_list_t& outs) {
  int b, c, n, npoints, nsample;
  SSAttrs(attr)
      .get<int>("b", b)
      .get<int>("c", c)
      .get<int>("n", n)
      .get<int>("npoints", npoints)
      .get<int>("nsample", nsample)
      .done();

  at::Tensor points_tensor, idx_tensor, out_tensor;
  points_tensor = buildATensor(ctx, ins[0]);
  idx_tensor = buildATensor(ctx, ins[1]);
  out_tensor = buildATensor(ctx, outs[0]);
  group_points(b, c, n, npoints, nsample, points_tensor, idx_tensor,
               out_tensor);
}

void group_points_backward_parrots(CudaContext& ctx, const SSElement& attr,
                                   const OperatorBase::in_list_t& ins,
                                   OperatorBase::out_list_t& outs) {
  int b, c, n, npoints, nsample;
  SSAttrs(attr)
      .get<int>("b", b)
      .get<int>("c", c)
      .get<int>("n", n)
      .get<int>("npoints", npoints)
      .get<int>("nsample", nsample)
      .done();

  at::Tensor grad_out_tensor, idx_tensor, grad_points_tensor;
  grad_out_tensor = buildATensor(ctx, ins[0]);
  idx_tensor = buildATensor(ctx, ins[1]);
  grad_points_tensor = buildATensor(ctx, outs[0]);
  group_points_backward(b, c, n, npoints, nsample, grad_out_tensor, idx_tensor,
                        grad_points_tensor);
}

PARROTS_EXTENSION_REGISTER(group_points)
    .attr("b")
    .attr("c")
    .attr("n")
    .attr("npoints")
    .attr("nsample")
    .input(2)
    .output(1)
#ifdef MMCV_WITH_CUDA
    .apply(group_points_parrots)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(group_points_backward)
    .attr("b")
    .attr("c")
    .attr("n")
    .attr("npoints")
    .attr("nsample")
    .input(2)
    .output(1)
#ifdef MMCV_WITH_CUDA
    .apply(group_points_backward_parrots)
#endif
    .done();
