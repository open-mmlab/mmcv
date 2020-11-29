// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_cpu.cpp
#include "box_iou_rotated_utils.hpp"
#include "parrots_cpp_helper.hpp"

template <typename T>
void box_iou_rotated_cpu_kernel(const DArrayLite boxes1, const DArrayLite boxes2,
                                DArrayLite ious) {

  int num_boxes1 = boxes1.dim(0);
  int num_boxes2 = boxes2.dim(0);

  auto ious_ptr = ious.ptr<float>();

  for (int i = 0; i < num_boxes1; i++) {
    for (int j = 0; j < num_boxes2; j++) {
      ious_ptr[i * num_boxes2 + j] = single_box_iou_rotated<T>(boxes1[i].ptr<T>(), boxes2[j].ptr<T>());
    }
  }
}

DArrayLite box_iou_rotated_cpu_launcher(const DArrayLite boxes1, const DArrayLite boxes2, HostContext& ctx) {
  int num_boxes1 = boxes1.dim(0);
  int num_boxes2 = boxes2.dim(0);

  auto ious = ctx.createDArrayLite(
      DArraySpec::array(Prim::Float32, DArrayShape(num_boxes1 * num_boxes2)));

  box_iou_rotated_cpu_kernel<float>(boxes1, boxes2, ious);

  // reshape from 1d array to 2d array
  return ious.view({num_boxes1, num_boxes2});
}
