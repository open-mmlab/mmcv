#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>
#include <parrots/compute/aten.hpp>
#include "furthest_point_sample_ext_pytorch.h"

using namespace parrots;

template<typename T>
void furthest_point_sampling_parrots(T& ctx, const SSElement& attr,
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

  furthest_point_sampling(b, n, m, points_tensor, temp_tensor, idx_tensor);
}

template<typename T>
void furthest_point_sampling_with_dist_parrots(T& ctx, const SSElement& attr,
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
  
  furthest_point_sampling_with_dist(b, n, m, points_tensor, temp_tensor, idx_tensor);
}

PARROTS_EXTENSION_REGISTER(furthest_point_sampling)
    .attr("b")
    .attr("n")
    .attr("m")
    .input(1)
    .output(2)
    .apply(furthest_point_sampling_parrots<HostContext>)
#ifdef MMCV_WITH_CUDA
    .apply(furthest_point_sampling_parrots<CudaContext>)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(furthest_point_sampling_with_dist)
    .attr("b")
    .attr("n")
    .attr("m")
    .input(1)
    .output(2)
    .apply(furthest_point_sampling_with_dist_parrots<HostContext>)
#ifdef MMCV_WITH_CUDA
    .apply(furthest_point_sampling_with_dist_parrots<CudaContext>)
#endif
    .done();

