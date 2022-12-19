
#include <diopi/functions.h>
#include <torch/nn.h>
#include <torch/optim.h>
#include <iostream>
#include <math.h>

#include "helper.hpp"

#define FLT_MIN		__FLT_MIN__

using namespace at;

Tensor NMSCUDAKernelLauncher(Tensor boxes, Tensor scores, float iou_threshold,
                             int offset);

diopiError_t diopiNms(diopiContextHandle_t ctx, diopiTensorHandle_t* out, const diopiTensorHandle_t dets,
        const diopiTensorHandle_t scores, double iouThreshold, int64_t offset) {
    auto atDets = ::impl::aten::buildATen(dets);
    auto atScores = ::impl::aten::buildATen(scores);
    auto atOut = NMSCUDAKernelLauncher(atDets, atScores, iouThreshold, offset);
    ::impl::aten::buildDiopiTensor(ctx, atOut, out);
}

void ChamferDistanceForwardCUDAKernelLauncher(
    const Tensor xyz1, const Tensor xyz2, const Tensor dist1,
    const Tensor dist2, const Tensor idx1, const Tensor idx2);

void ChamferDistanceBackwardCUDAKernelLauncher(
    const Tensor xyz1, const Tensor xyz2, Tensor idx1, Tensor idx2,
    Tensor grad_dist1, Tensor grad_dist2, Tensor grad_xyz1, Tensor grad_xyz2);

diopiError_t diopiChamferDistance(diopiContextHandle_t ctx, const diopiTensorHandle_t xyz1,
                                  const diopiTensorHandle_t xyz2, diopiTensorHandle_t dist1, diopiTensorHandle_t dist2,
                                  diopiTensorHandle_t idx1, diopiTensorHandle_t idx2) {
    auto xyz1_in = ::impl::aten::buildATen(xyz1);
    auto xyz2_in = ::impl::aten::buildATen(xyz2);
    auto dist1_out = ::impl::aten::buildATen(dist1);
    auto dist2_out = ::impl::aten::buildATen(dist2);
    auto idx1_out = ::impl::aten::buildATen(idx1);
    auto idx2_out = ::impl::aten::buildATen(idx2);
    ChamferDistanceForwardCUDAKernelLauncher(
        xyz1_in, xyz2_in, dist1_out, dist2_out, idx1_out, idx2_out);
}

diopiError_t diopiChamferDistanceBackward(diopiContextHandle_t ctx, const diopiTensorHandle_t xyz1, const diopiTensorHandle_t xyz2,
                                            const diopiTensorHandle_t idx1, const diopiTensorHandle_t idx2, const diopiTensorHandle_t grad_dist1, const diopiTensorHandle_t grad_dist2,
                                            diopiTensorHandle_t grad_xyz1, diopiTensorHandle_t grad_xyz2) {
    auto xyz1_in = ::impl::aten::buildATen(xyz1);
    auto xyz2_in = ::impl::aten::buildATen(xyz2);
    auto idx1_in = ::impl::aten::buildATen(idx1);
    auto idx2_in = ::impl::aten::buildATen(idx2);
    auto grad_dist1_in = ::impl::aten::buildATen(grad_dist1);
    auto grad_dist2_in = ::impl::aten::buildATen(grad_dist2);
    auto grad_xyz1_out = ::impl::aten::buildATen(grad_xyz1);
    auto grad_xyz2_out = ::impl::aten::buildATen(grad_xyz2);
    ChamferDistanceBackwardCUDAKernelLauncher(
        xyz1_in, xyz2_in, idx1_in, idx2_in, grad_dist1_in, grad_dist2_in, grad_xyz1_out, grad_xyz2_out);
}

void PrROIPoolForwardCUDAKernelLauncher(Tensor input, Tensor rois,
                                        Tensor output, int pooled_height,
                                        int pooled_width, float spatial_scale);

void PrROIPoolBackwardCUDAKernelLauncher(Tensor grad_output, Tensor rois,
                                         Tensor grad_input, int pooled_height,
                                         int pooled_width, float spatial_scale);

void PrROIPoolCoorBackwardCUDAKernelLauncher(
    Tensor output, Tensor grad_output, Tensor input, Tensor rois,
    Tensor grad_rois, int pooled_height, int pooled_width, float spatial_scale);

diopiError_t diopiPrroiPool(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t rois, diopiTensorHandle_t output,
                                      int64_t pooled_height, int64_t pooled_width, float spatial_scale) {
    auto input_in = ::impl::aten::buildATen(input);
    auto rois_in = ::impl::aten::buildATen(rois);
    auto output_out = ::impl::aten::buildATen(output);
    PrROIPoolForwardCUDAKernelLauncher(input_in, rois_in, output_out, pooled_height, pooled_width, spatial_scale);

}

diopiError_t diopiPrroiPoolbackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output, diopiTensorHandle_t rois, diopiTensorHandle_t grad_input,
                                              int64_t pooled_height, int64_t pooled_width, float spatial_scale) {
    auto grad_output_in = ::impl::aten::buildATen(grad_output);
    auto rois_in = ::impl::aten::buildATen(rois);
    auto grad_input_out = ::impl::aten::buildATen(grad_input);
    PrROIPoolBackwardCUDAKernelLauncher(grad_output_in, rois_in, grad_input_out, pooled_height, pooled_width, spatial_scale);
}

diopiError_t diopiPrroiPoolCoorBackward(diopiContextHandle_t ctx, diopiTensorHandle_t output, diopiTensorHandle_t grad_output, diopiTensorHandle_t input, diopiTensorHandle_t rois,
                                        diopiTensorHandle_t grad_rois, int64_t pooled_height, int64_t pooled_width, float spatial_scale) {
    auto output_in = ::impl::aten::buildATen(output);
    auto grad_output_in = ::impl::aten::buildATen(grad_output);
    auto input_in = ::impl::aten::buildATen(input);
    auto rois_in = ::impl::aten::buildATen(rois);
    auto grad_rois_out = ::impl::aten::buildATen(grad_rois);
    PrROIPoolCoorBackwardCUDAKernelLauncher(output_in, grad_output_in, input_in, rois_in, grad_rois_out, pooled_height, pooled_width, spatial_scale);
}

void ActiveRotatedFilterForwardCUDAKernelLauncher(const Tensor input,
                                                  const Tensor indices,
                                                  Tensor output);

void ActiveRotatedFilterBackwardCUDAKernelLauncher(const Tensor grad_out,
                                                   const Tensor indices,
                                                   Tensor grad_in);

diopiError_t diopiActiveRotatedFilter(diopiContextHandle_t ctx, const diopiTensorHandle_t input, const diopiTensorHandle_t indices, diopiTensorHandle_t output) {
    auto input_in = ::impl::aten::buildATen(input);
    auto indices_in = ::impl::aten::buildATen(indices);
    auto output_out = ::impl::aten::buildATen(output);
    ActiveRotatedFilterForwardCUDAKernelLauncher(input_in, indices_in, output_out);
}

diopiError_t diopiActiveRotatedFilterBackward(diopiContextHandle_t ctx, const diopiTensorHandle_t grad_out, const diopiTensorHandle_t indices, diopiTensorHandle_t grad_in) {
    auto grad_out_in = ::impl::aten::buildATen(grad_out);
    auto indices_in = ::impl::aten::buildATen(indices);
    auto grad_in_out = ::impl::aten::buildATen(grad_in);
    ActiveRotatedFilterBackwardCUDAKernelLauncher(grad_out_in, indices_in, grad_in_out);
}

void AssignScoreWithKForwardCUDAKernelLauncher(
    int B, int N0, int N1, int M, int K, int O, int aggregate,
    const Tensor& points, const Tensor& centers, const Tensor& scores,
    const Tensor& knn_idx, Tensor& output);

void AssignScoreWithKBackwardCUDAKernelLauncher(
    int B, int N0, int N1, int M, int K, int O, int aggregate,
    const Tensor& grad_out, const Tensor& points, const Tensor& centers,
    const Tensor& scores, const Tensor& knn_idx, Tensor& grad_points,
    Tensor& grad_centers, Tensor& grad_scores);

diopiError_t diopiAssignScoreWithk(diopiContextHandle_t ctx, const diopiTensorHandle_t points, const diopiTensorHandle_t centers,
                                const diopiTensorHandle_t scores, const diopiTensorHandle_t knn_idx,
                                diopiTensorHandle_t output, int64_t B, int64_t N0, int64_t N1, int64_t M,
                                int64_t K, int64_t O, int64_t aggregate) {
    auto points_in = ::impl::aten::buildATen(points);
    auto centers_in = ::impl::aten::buildATen(centers);
    auto scores_in = ::impl::aten::buildATen(scores);
    auto knn_idx_in = ::impl::aten::buildATen(knn_idx);
    auto output_out = ::impl::aten::buildATen(output);
    AssignScoreWithKForwardCUDAKernelLauncher(
      B, N0, N1, M, K, O, aggregate, points_in, centers_in, scores_in, knn_idx_in, output_out);
}

diopiError_t diopiAssignScoreWithkBackward(diopiContextHandle_t ctx, const diopiTensorHandle_t grad_out, const diopiTensorHandle_t points,
                                 const diopiTensorHandle_t centers, const diopiTensorHandle_t scores,
                                 const diopiTensorHandle_t knn_idx, diopiTensorHandle_t grad_points,
                                 diopiTensorHandle_t grad_centers, diopiTensorHandle_t grad_scores,
                                 int64_t B, int64_t N0, int64_t N1, int64_t M, int64_t K, int64_t O,
                                 int64_t aggregate) {
    auto grad_out_in = ::impl::aten::buildATen(grad_out);
    auto points_in = ::impl::aten::buildATen(points);
    auto centers_in = ::impl::aten::buildATen(centers);
    auto scores_in = ::impl::aten::buildATen(scores);
    auto knn_idx_in = ::impl::aten::buildATen(knn_idx);
    auto grad_points_out = ::impl::aten::buildATen(grad_points);
    auto grad_centers_out = ::impl::aten::buildATen(grad_centers);
    auto grad_scores_out = ::impl::aten::buildATen(grad_scores);
    AssignScoreWithKBackwardCUDAKernelLauncher(
      B, N0, N1, M, K, O, aggregate, grad_out_in, points_in, centers_in, scores_in, knn_idx_in,
      grad_points_out, grad_centers_out, grad_scores_out);
}

void BBoxOverlapsCUDAKernelLauncher(const Tensor bboxes1, const Tensor bboxes2,
                                    Tensor ious, const int mode,
                                    const bool aligned, const int offset);

diopiError_t diopiBboxOverlaps(diopiContextHandle_t ctx, const diopiTensorHandle_t bboxes1, const diopiTensorHandle_t bboxes2, diopiTensorHandle_t ious,
                        const int64_t mode, const bool aligned, const int64_t offset) {
    auto bboxes1_in = ::impl::aten::buildATen(bboxes1);
    auto bboxes2_in = ::impl::aten::buildATen(bboxes2);
    auto ious_out = ::impl::aten::buildATen(ious);
    BBoxOverlapsCUDAKernelLauncher(bboxes1_in, bboxes2_in, ious_out, mode, aligned, offset);
}

void BorderAlignForwardCUDAKernelLauncher(const Tensor& input,
                                          const Tensor& boxes, Tensor output,
                                          Tensor argmax_idx,
                                          const int pool_size);

void BorderAlignBackwardCUDAKernelLauncher(const Tensor& grad_output,
                                           const Tensor& boxes,
                                           const Tensor& argmax_idx,
                                           Tensor grad_input,
                                           const int pool_size);

diopiError_t diopiBorderAlign(diopiContextHandle_t ctx, const diopiTensorHandle_t input, const diopiTensorHandle_t boxes,
                               diopiTensorHandle_t output, diopiTensorHandle_t argmax_idx,
                               const int64_t pool_size) {
    auto input_in = ::impl::aten::buildATen(input);
    auto boxes_in = ::impl::aten::buildATen(boxes);
    auto output_out = ::impl::aten::buildATen(output);
    auto argmax_idx_out = ::impl::aten::buildATen(argmax_idx);
    BorderAlignForwardCUDAKernelLauncher(input_in, boxes_in, output_out, argmax_idx_out,
                                         pool_size);
}

diopiError_t diopiBorderAlignBackward(diopiContextHandle_t ctx, const diopiTensorHandle_t grad_output, const diopiTensorHandle_t boxes,
                                const diopiTensorHandle_t argmax_idx, diopiTensorHandle_t grad_input,
                                const int64_t pool_size) {
    auto grad_output_in = ::impl::aten::buildATen(grad_output);
    auto boxes_in = ::impl::aten::buildATen(boxes);
    auto argmax_idx_in = ::impl::aten::buildATen(argmax_idx);
    auto grad_input_out = ::impl::aten::buildATen(grad_input);
    BorderAlignBackwardCUDAKernelLauncher(grad_output_in, boxes_in, argmax_idx_in,
                                          grad_input_out, pool_size);
}

void ConvexIoUCUDAKernelLauncher(const Tensor pointsets, const Tensor polygons,
                                 Tensor ious);

void ConvexGIoUCUDAKernelLauncher(const Tensor pointsets, const Tensor polygons,
                                  Tensor output);

diopiError_t diopiConvexIou(diopiContextHandle_t ctx, const diopiTensorHandle_t pointsets, const diopiTensorHandle_t polygons, diopiTensorHandle_t ious) {
    auto pointsets_in = ::impl::aten::buildATen(pointsets);
    auto polygons_in = ::impl::aten::buildATen(polygons);
    auto ious_out = ::impl::aten::buildATen(ious);
    ConvexIoUCUDAKernelLauncher(pointsets_in, polygons_in, ious_out);
}

diopiError_t diopiConvexGiou(diopiContextHandle_t ctx, const diopiTensorHandle_t pointsets, const diopiTensorHandle_t polygons, diopiTensorHandle_t output) {
    auto pointsets_in = ::impl::aten::buildATen(pointsets);
    auto polygons_in = ::impl::aten::buildATen(polygons);
    auto output_out = ::impl::aten::buildATen(output);
    ConvexGIoUCUDAKernelLauncher(pointsets_in, polygons_in, output_out);
}

void CorrelationForwardCUDAKernelLauncher(Tensor input1, Tensor input2,
                                          Tensor output, int kH, int kW,
                                          int patchH, int patchW, int padH,
                                          int padW, int dilationH,
                                          int dilationW, int dilation_patchH,
                                          int dilation_patchW, int dH, int dW);

void CorrelationBackwardCUDAKernelLauncher(Tensor grad_output, Tensor input1,
                                           Tensor input2, Tensor grad_input1,
                                           Tensor grad_input2, int kH, int kW,
                                           int patchH, int patchW, int padH,
                                           int padW, int dilationH,
                                           int dilationW, int dilation_patchH,
                                           int dilation_patchW, int dH, int dW);

diopiError_t diopiCorrelation(diopiContextHandle_t ctx, diopiTensorHandle_t input1, diopiTensorHandle_t input2, diopiTensorHandle_t output, int64_t kH,
                         int64_t kW, int64_t patchH, int64_t patchW, int64_t padH, int64_t padW,
                         int64_t dilationH, int64_t dilationW, int64_t dilation_patchH,
                         int64_t dilation_patchW, int64_t dH, int64_t dW) {
    auto input1_in = ::impl::aten::buildATen(input1);
    auto input2_in = ::impl::aten::buildATen(input2);
    auto output_out = ::impl::aten::buildATen(output);
    CorrelationForwardCUDAKernelLauncher(
      input1_in, input2_in, output_out, kH, kW, patchH, patchW, padH, padW, dilationH,
      dilationW, dilation_patchH, dilation_patchW, dH, dW);
}

diopiError_t diopiCorrelationBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output, diopiTensorHandle_t input1, diopiTensorHandle_t input2,
                          diopiTensorHandle_t grad_input1, diopiTensorHandle_t grad_input2, int64_t kH,
                          int64_t kW, int64_t patchH, int64_t patchW, int64_t padH, int64_t padW,
                          int64_t dilationH, int64_t dilationW, int64_t dilation_patchH,
                          int64_t dilation_patchW, int64_t dH, int64_t dW) {
    auto grad_output_in = ::impl::aten::buildATen(grad_output);
    auto input1_in = ::impl::aten::buildATen(input1);
    auto input2_in = ::impl::aten::buildATen(input2);
    auto grad_input1_out = ::impl::aten::buildATen(grad_input1);
    auto grad_input2_out = ::impl::aten::buildATen(grad_input2);
    CorrelationBackwardCUDAKernelLauncher(
      grad_output_in, input1_in, input2_in, grad_input1_out, grad_input2_out, kH, kW, patchH,
      patchW, padH, padW, dilationH, dilationW, dilation_patchH,
      dilation_patchW, dH, dW);
}
