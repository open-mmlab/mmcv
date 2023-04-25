// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "assign_score_withk_pytorch.h"

#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include <parrots/diopi.hpp>
#endif

using namespace parrots;

#ifdef MMCV_WITH_CUDA
#ifdef MMCV_WITH_DIOPI
void assign_score_withk_forward_cuda_parrots_diopi(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  int B, N0, N1, M, K, O, aggregate;
  SSAttrs(attr)
      .get<int>("B", B)
      .get<int>("N0", N0)
      .get<int>("N1", N1)
      .get<int>("M", M)
      .get<int>("K", K)
      .get<int>("O", O)
      .get<int>("aggregate", aggregate)
      .done();

  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto points =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));
  auto centers =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[1]));
  auto scores =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[2]));
  auto knn_idx =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[3]));

  auto output = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);
  PARROTS_CALLDIOPI(diopiAssignScoreWithk(ch, points, centers, scores, knn_idx,
                                          output, B, N0, N1, M, K, O,
                                          aggregate));
}

void assign_score_withk_backward_cuda_parrots_diopi(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  int B, N0, N1, M, K, O, aggregate;
  SSAttrs(attr)
      .get<int>("B", B)
      .get<int>("N0", N0)
      .get<int>("N1", N1)
      .get<int>("M", M)
      .get<int>("K", K)
      .get<int>("O", O)
      .get<int>("aggregate", aggregate)
      .done();

  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto grad_out =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));
  auto points =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[1]));
  auto centers =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[2]));
  auto scores =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[3]));
  auto knn_idx =
      reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[4]));

  auto grad_points = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);
  auto grad_centers = reinterpret_cast<diopiTensorHandle_t>(&outs[1]);
  auto grad_scores = reinterpret_cast<diopiTensorHandle_t>(&outs[2]);
  PARROTS_CALLDIOPI(diopiAssignScoreWithkBackward(
      ch, grad_out, points, centers, scores, knn_idx, grad_points, grad_centers,
      grad_scores, B, N0, N1, M, K, O, aggregate));
}
#else
void assign_score_withk_forward_cuda_parrots(CudaContext& ctx,
                                             const SSElement& attr,
                                             const OperatorBase::in_list_t& ins,
                                             OperatorBase::out_list_t& outs) {
  int B, N0, N1, M, K, O, aggregate;
  SSAttrs(attr)
      .get<int>("B", B)
      .get<int>("N0", N0)
      .get<int>("N1", N1)
      .get<int>("M", M)
      .get<int>("K", K)
      .get<int>("O", O)
      .get<int>("aggregate", aggregate)
      .done();

  const auto& points = buildATensor(ctx, ins[0]);
  const auto& centers = buildATensor(ctx, ins[1]);
  const auto& scores = buildATensor(ctx, ins[2]);
  const auto& knn_idx = buildATensor(ctx, ins[3]);

  auto output = buildATensor(ctx, outs[0]);
  assign_score_withk_forward(points, centers, scores, knn_idx, output, B, N0,
                             N1, M, K, O, aggregate);
}

void assign_score_withk_backward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  int B, N0, N1, M, K, O, aggregate;
  SSAttrs(attr)
      .get<int>("B", B)
      .get<int>("N0", N0)
      .get<int>("N1", N1)
      .get<int>("M", M)
      .get<int>("K", K)
      .get<int>("O", O)
      .get<int>("aggregate", aggregate)
      .done();

  const auto& grad_out = buildATensor(ctx, ins[0]);
  const auto& points = buildATensor(ctx, ins[1]);
  const auto& centers = buildATensor(ctx, ins[2]);
  const auto& scores = buildATensor(ctx, ins[3]);
  const auto& knn_idx = buildATensor(ctx, ins[4]);

  auto grad_points = buildATensor(ctx, outs[0]);
  auto grad_centers = buildATensor(ctx, outs[1]);
  auto grad_scores = buildATensor(ctx, outs[2]);
  assign_score_withk_backward(grad_out, points, centers, scores, knn_idx,
                              grad_points, grad_centers, grad_scores, B, N0, N1,
                              M, K, O, aggregate);
}
#endif

PARROTS_EXTENSION_REGISTER(assign_score_withk_forward)
    .attr("B")
    .attr("N0")
    .attr("N1")
    .attr("M")
    .attr("K")
    .attr("O")
    .attr("aggregate")
    .input(4)
    .output(1)
#ifdef MMCV_WITH_DIOPI
    .apply(assign_score_withk_forward_cuda_parrots_diopi)
#else
    .apply(assign_score_withk_forward_cuda_parrots)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(assign_score_withk_backward)
    .attr("B")
    .attr("N0")
    .attr("N1")
    .attr("M")
    .attr("K")
    .attr("O")
    .attr("aggregate")
    .input(5)
    .output(3)
#ifdef MMCV_WITH_DIOPI
    .apply(assign_score_withk_backward_cuda_parrots_diopi)
#else
    .apply(assign_score_withk_backward_cuda_parrots)
#endif
    .done();
#endif
