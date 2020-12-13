#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>
#include <parrots/compute/aten.hpp>
#include "furthest_point_sample_ext_pytorch.h"

using namespace parrots;

/*
int furthest_point_sampling_wrapper(int b, int n, int m,
                                    at::Tensor points_tensor,
                                    at::Tensor temp_tensor,
                                    at::Tensor idx_tensor);


int furthest_point_sampling_with_dist_wrapper(int b, int n, int m,
                                              at::Tensor points_tensor,
                                              at::Tensor temp_tensor,
                                              at::Tensor idx_tensor);
*/

void furthest_point_sampling_parrots(CudaContext& ctx, const SSElement& attr,
                        const OperatorBase::in_list_t& ins,
                        OperatorBase::out_list_t& outs){
  int b, n, m;
  SSAttrs(attr)
     .get("b", b)
     .get("n", n)
     .get("m", m)
     .done();

  at::Tensor points_tensor, temp_tensor, idx_tensor;
  points_tensor = buildATensor(ctx, ins[0]);
  temp_tensor = buildATensor(ctx, outs[0]);
  idx_tensor = buildATensor(ctx, outs[1]);

  furthest_point_sampling_wrapper(b, n, m, points_tensor, temp_tensor, idx_tensor);
}

void furthest_point_sampling_with_dist_parrots(CudaContext& ctx, const SSElement& attr,
                        const OperatorBase::in_list_t& ins,
                        OperatorBase::out_list_t& outs){
  int b, n, m;
  SSAttrs(attr)
     .get("b", b)
     .get("n", n)
     .get("m", m)
     .done();

  at::Tensor points_tensor, temp_tensor, idx_tensor;
  points_tensor = buildATensor(ctx, ins[0]);
  temp_tensor = buildATensor(ctx, outs[0]);
  idx_tensor = buildATensor(ctx, outs[1]);
  
  furthest_point_sampling_with_dist_wrapper(b, n, m, points_tensor, temp_tensor, idx_tensor);
}

PARROTS_EXTENSION_REGISTER(furthest_point_sampling_wrapper)
    .attr("b")
    .attr("n")
    .attr("m")
    .input(1)
    .output(2)
    .apply(furthest_point_sampling_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(furthest_point_sampling_with_dist_wrapper)
    .attr("b")
    .attr("n")
    .attr("m")
    .input(1)
    .output(2)
    .apply(furthest_point_sampling_with_dist_parrots)
    .done();

