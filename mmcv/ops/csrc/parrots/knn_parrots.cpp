// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "knn_pytorch.h"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <parrots/diopi.hpp>
#endif

using namespace parrots;

#ifdef MMCV_WITH_CUDA
#ifdef MMCV_WITH_DIOPI
void knn_forward_cuda_parrots_diopi(CudaContext& ctx, const SSElement& attr,
                              const OperatorBase::in_list_t& ins,
                              OperatorBase::out_list_t& outs) {
  int b, n, m, nsample;
  SSAttrs(attr)
      .get<int>("b", b)
      .get<int>("n", n)
      .get<int>("m", m)
      .get<int>("nsample", nsample)
      .done();

  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto xyz_tensor = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));
  auto new_xyz_tensor = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[1]));

  auto idx_tensor = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);
  auto dist2_tensor = reinterpret_cast<diopiTensorHandle_t>(&outs[1]);

  PARROTS_CALLDIOPI(diopiKnn(ch, xyz_tensor, new_xyz_tensor, idx_tensor, dist2_tensor, b, n, m, nsample));
}
#else
void knn_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                              const OperatorBase::in_list_t& ins,
                              OperatorBase::out_list_t& outs) {
  int b, n, m, nsample;
  SSAttrs(attr)
      .get<int>("b", b)
      .get<int>("n", n)
      .get<int>("m", m)
      .get<int>("nsample", nsample)
      .done();

  auto xyz_tensor = buildATensor(ctx, ins[0]);
  auto new_xyz_tensor = buildATensor(ctx, ins[1]);

  auto idx_tensor = buildATensor(ctx, outs[0]);
  auto dist2_tensor = buildATensor(ctx, outs[1]);

  knn_forward(xyz_tensor, new_xyz_tensor, idx_tensor, dist2_tensor, b, n, m,
              nsample);
}

#endif
PARROTS_EXTENSION_REGISTER(knn_forward)
    .attr("b")
    .attr("n")
    .attr("m")
    .attr("nsample")
    .input(2)
    .output(2)
#ifdef MMCV_WITH_DIOPI
    .apply(knn_forward_cuda_parrots_diopi)
#else
    .apply(knn_forward_cuda_parrots)
#endif
    .done();
#endif
