#include <diopi/functions.h>

#include "helper.hpp"

using namespace at;

void BBoxOverlapsMLUKernelLauncher(const Tensor bboxes1, const Tensor bboxes2,
                                   Tensor ious, const int32_t mode,
                                   const bool aligned, const int32_t offset);

diopiError_t diopiBboxOverlaps(diopiContextHandle_t ctx,
                               diopiConstTensorHandle_t bboxes1,
                               diopiConstTensorHandle_t bboxes2,
                               diopiTensorHandle_t ious, const int64_t mode,
                               const bool aligned, const int64_t offset) {
  auto bboxes1_in = ::camb::aten::buildATen(bboxes1);
  auto bboxes2_in = ::camb::aten::buildATen(bboxes2);
  auto ious_out = ::camb::aten::buildATen(ious);
  BBoxOverlapsMLUKernelLauncher(bboxes1_in, bboxes2_in, ious_out, mode, aligned, offset);
}

Tensor NMSMLUKernelLauncher(Tensor boxes, Tensor scores, float iou_threshold, int offset);

diopiError_t diopiNmsMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t *out,
                      diopiConstTensorHandle_t dets,
                      diopiConstTensorHandle_t scores, double iouThreshold,
                      int64_t offset) {
  auto atDets = ::camb::aten::buildATen(dets);
  auto atScores = ::camb::aten::buildATen(scores);
  auto atOut = NMSMLUKernelLauncher(atDets, atScores, iouThreshold, offset);
  ::camb::aten::buildDiopiTensor(ctx, atOut, out);
}
