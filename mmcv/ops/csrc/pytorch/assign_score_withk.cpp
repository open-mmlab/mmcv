// Modified from
// https://github.com/CVMI-Lab/PAConv/tree/main/scene_seg/lib/paconv_lib/src/gpu
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>
#include "diopi.hpp"
#endif

void assign_score_withk_forward_impl(int B, int N0, int N1, int M, int K, int O,
                                     int aggregate, const Tensor& points,
                                     const Tensor& centers,
                                     const Tensor& scores,
                                     const Tensor& knn_idx, Tensor& output) {
  DISPATCH_DEVICE_IMPL(assign_score_withk_forward_impl, B, N0, N1, M, K, O,
                       aggregate, points, centers, scores, knn_idx, output);
}

void assign_score_withk_backward_impl(
    int B, int N0, int N1, int M, int K, int O, int aggregate,
    const Tensor& grad_out, const Tensor& points, const Tensor& centers,
    const Tensor& scores, const Tensor& knn_idx, Tensor& grad_points,
    Tensor& grad_centers, Tensor& grad_scores) {
  DISPATCH_DEVICE_IMPL(assign_score_withk_backward_impl, B, N0, N1, M, K, O,
                       aggregate, grad_out, points, centers, scores, knn_idx,
                       grad_points, grad_centers, grad_scores);
}

void assign_score_withk_forward(const Tensor& points, const Tensor& centers,
                                const Tensor& scores, const Tensor& knn_idx,
                                Tensor& output, int B, int N0, int N1, int M,
                                int K, int O, int aggregate) {
#ifdef MMCV_WITH_DIOPI
  auto points_p = toDiopiTensorHandle(points);
  diopiDevice_t device;
  diopiGetTensorDevice(points_p, &device);
  if (device == diopi_host) {
      assign_score_withk_forward_impl(B, N0, N1, M, K, O, aggregate, points,
                                  centers, scores, knn_idx, output);
      return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto centers_p = toDiopiTensorHandle(centers);
  auto scores_p = toDiopiTensorHandle(scores);
  auto knn_idx_p = toDiopiTensorHandle(knn_idx);
  auto output_p = toDiopiTensorHandle(output);
  if(&diopiAssignScoreWithk) {
   diopiAssignScoreWithk(ch, points_p, centers_p, scores_p, knn_idx_p, output_p, B, N0,
                                          N1, M, K, O, aggregate);
  } else {
   assign_score_withk_forward_impl(B, N0, N1, M, K, O, aggregate, points,
                                  centers, scores, knn_idx, output);
  }
#else
  assign_score_withk_forward_impl(B, N0, N1, M, K, O, aggregate, points,
                                  centers, scores, knn_idx, output);
#endif
}

void assign_score_withk_backward(const Tensor& grad_out, const Tensor& points,
                                 const Tensor& centers, const Tensor& scores,
                                 const Tensor& knn_idx, Tensor& grad_points,
                                 Tensor& grad_centers, Tensor& grad_scores,
                                 int B, int N0, int N1, int M, int K, int O,
                                 int aggregate) {
#ifdef MMCV_WITH_DIOPI
  auto grad_out_p = toDiopiTensorHandle(grad_out);
  diopiDevice_t device;
  diopiGetTensorDevice(grad_out_p, &device);
  if (device == diopi_host) {
      assign_score_withk_backward_impl(B, N0, N1, M, K, O, aggregate, grad_out,
                                   points, centers, scores, knn_idx,
                                   grad_points, grad_centers, grad_scores);
      return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto points_p = toDiopiTensorHandle(points);
  auto centers_p = toDiopiTensorHandle(centers);
  auto scores_p = toDiopiTensorHandle(scores);
  auto knn_idx_p = toDiopiTensorHandle(knn_idx);
  auto grad_points_p = toDiopiTensorHandle(grad_points);
  auto grad_centers_p = toDiopiTensorHandle(grad_centers);
  auto grad_scores_p = toDiopiTensorHandle(grad_scores);
  if (&diopiAssignScoreWithkBackward) {
      diopiAssignScoreWithkBackward(ch, grad_out_p, points_p, centers_p, scores_p, knn_idx_p,
                                    grad_points_p, grad_centers_p, grad_scores_p, B, N0, N1,
                                    M, K, O, aggregate);
  }
  else {
      assign_score_withk_backward_impl(B, N0, N1, M, K, O, aggregate, grad_out,
                                       points, centers, scores, knn_idx,
                                       grad_points, grad_centers, grad_scores);
  }
#else
  assign_score_withk_backward_impl(B, N0, N1, M, K, O, aggregate, grad_out,
                                   points, centers, scores, knn_idx,
                                   grad_points, grad_centers, grad_scores);
#endif
}
