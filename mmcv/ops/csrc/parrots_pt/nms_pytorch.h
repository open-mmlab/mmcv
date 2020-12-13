#ifndef NMS_PYTORCH_H
#define NMS_PYTORCH_H
#include <torch/extension.h>
using namespace at;

Tensor nms(Tensor boxes, Tensor scores, float iou_threshold, int offset);

Tensor softnms(Tensor boxes, Tensor scores, Tensor dets, float iou_threshold,
               float sigma, float min_score, int method, int offset);

std::vector<std::vector<int> > nms_match(Tensor dets, float iou_threshold);

Tensor nms_rotated(const Tensor dets, const Tensor scores, const Tensor order,
                   const Tensor dets_sorted, const float iou_threshold,
                   const int multi_label);

#endif  //NMS_PYTORCH_H
