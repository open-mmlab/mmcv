// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_cpu.cpp
#include "box_iou_rotated_utils.hpp"
#include "pytorch_cpp_helper.hpp"

template <typename T>
void box_iou_rotated_cpu_kernel(const Tensor boxes1, const Tensor boxes2,
                                Tensor ious) {
  auto widths1 = boxes1.select(1, 2).contiguous();
  auto heights1 = boxes1.select(1, 3).contiguous();
  auto widths2 = boxes2.select(1, 2).contiguous();
  auto heights2 = boxes2.select(1, 3).contiguous();

  Tensor areas1 = widths1 * heights1;
  Tensor areas2 = widths2 * heights2;

  auto num_boxes1 = boxes1.size(0);
  auto num_boxes2 = boxes2.size(0);

  for (int i = 0; i < num_boxes1; i++) {
    for (int j = 0; j < num_boxes2; j++) {
      ious[i * num_boxes2 + j] = single_box_iou_rotated<T>(
          boxes1[i].data_ptr<T>(), boxes2[j].data_ptr<T>());
    }
  }
}

Tensor box_iou_rotated_cpu(const Tensor boxes1, const Tensor boxes2) {
  auto num_boxes1 = boxes1.size(0);
  auto num_boxes2 = boxes2.size(0);
  Tensor ious =
      at::empty({num_boxes1 * num_boxes2}, boxes1.options().dtype(at::kFloat));

  box_iou_rotated_cpu_kernel<float>(boxes1, boxes2, ious);

  // reshape from 1d array to 2d array
  auto shape = std::vector<int64_t>{num_boxes1, num_boxes2};
  return ious.reshape(shape);
}
