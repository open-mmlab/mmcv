// Copyright (c) OpenMMLab. All rights reserved
#ifndef IOU_3D_PYTORCH_H
#define IOU_3D_PYTORCH_H
#include <torch/types.h>
using namespace at;

void iou3d_boxes_iou_bev_forward(Tensor boxes_a, Tensor boxes_b,
                                 Tensor ans_iou);

void iou3d_nms_forward(Tensor boxes, Tensor keep, Tensor keep_num,
                       float nms_overlap_thresh);

void iou3d_nms_normal_forward(Tensor boxes, Tensor keep, Tensor keep_num,
                              float nms_overlap_thresh);

#endif  // IOU_3D_PYTORCH_H
