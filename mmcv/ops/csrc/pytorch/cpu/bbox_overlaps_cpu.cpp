// Copyright(c) OpenMMLab.All rights reserved.
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#define HOST_DEVICE

using torch::indexing::None;
using torch::indexing::Slice;

// template <typename T>
// HOST_DEVICE inline T single_box_iou(T const *const box1_raw,
//                                     T const *const box2_raw, const int
//                                     offset, const int mode_flag) {
//   auto box1_width = box1_raw[2] - box1_raw[0] + offset;
//   auto box1_height = box1_raw[3] - box1_raw[1] + offset;
//   auto box2_width = box2_raw[2] - box2_raw[0] + offset;
//   auto box2_height = box2_raw[3] - box2_raw[1] + offset;
//   const T area1 = box1_width * box1_height;
//   const T area2 = box2_width * box2_height;
//   if (area1 < 1e-14 || area2 < 1e-14) {
//     return 0.f;
//   }

//   // The smallest point on the right corner of the two boxes - the largest
//   point
//   // on the left corner of the two boxes is the width of the intersection W,
//   and
//   // the same is true for H
//   T inter_width =
//       std::min(box1_raw[2], box2_raw[2]) - std::max(box1_raw[0],
//       box2_raw[0]);
//   inter_width = std::max(inter_width + offset, 0.f);
//   T inter_height =
//       std::min(box1_raw[3], box2_raw[3]) - std::max(box1_raw[1],
//       box2_raw[1]);
//   inter_height = std::max(inter_height + offset, 0.f);
//   const T intersection = inter_width * inter_height;

//   T baseS = 1.0;
//   if (mode_flag == 0) {
//     baseS = (area1 + area2 - intersection);
//   } else if (mode_flag == 1) {
//     baseS = area1;
//   }
//   const T iou = intersection / baseS;
//   return iou;
// }

void bbox_overlaps_cpu_kernel(const Tensor boxes1, const Tensor boxes2,
                              Tensor ious, const int mode_flag,
                              const bool aligned, const int offset) {
  // int output_size = ious.numel();
  // auto num_boxes1 = boxes1.size(0);
  // auto num_boxes2 = boxes2.size(0);
  if (aligned) {
    Tensor lt = torch::max(boxes1.index({Slice(None), Slice({None, 2})}),
                           boxes2.index({Slice(None), Slice({None, 2})}));
    Tensor rb = torch::min(boxes1.index({Slice(None), Slice(2)}),
                           boxes2.index({Slice(None), Slice(2)}));
    Tensor wh = (rb - lt + offset).clamp(0.f, INT_MAX * 1.f);
    Tensor overlap = wh.index({Slice(None), 0}) * wh.index({Slice(None), 1});
    Tensor area1 = (boxes1.index({Slice(None), 2}) -
                    boxes1.index({Slice(None), 0}) + offset) *
                   (boxes1.index({Slice(None), 3}) -
                    boxes1.index({Slice(None), 1}) + offset);
    if (mode_flag == 0) {
      Tensor area2 = (boxes2.index({Slice(None), 2}) -
                      boxes2.index({Slice(None), 0}) + offset) *
                     (boxes2.index({Slice(None), 3}) -
                      boxes2.index({Slice(None), 1}) + offset);
      ious = overlap / (area1 + area2 - overlap);
    } else {
      ious = overlap / area1;
    }

    // #pragma omp parallel for

    // for (int i = 0; i < output_size; i++) {
    //   ious[i] = single_box_iou<T>(boxes1[i].data_ptr<T>(),
    //                               boxes2[i].data_ptr<T>(), offset,
    //                               mode_flag);
    // }
  } else {
    Tensor lt = torch::max(boxes1.index({Slice(None), None, Slice({None, 2})}),
                           boxes2.index({Slice(None), Slice({None, 2})}));
    Tensor rb = torch::min(boxes1.index({Slice(None), None, Slice(2)}),
                           boxes2.index({Slice(None), Slice(2)}));
    Tensor wh = (rb - lt + offset).clamp(0.f, INT_MAX * 1.f);
    Tensor overlap = wh.index({"...", 0}) * wh.index({"...", 1});
    Tensor area1 = (boxes1.index({Slice(None), 2}) -
                    boxes1.index({Slice(None), 0}) + offset) *
                   (boxes1.index({Slice(None), 3}) -
                    boxes1.index({Slice(None), 1}) + offset);
    if (mode_flag == 0) {
      Tensor area2 = (boxes2.index({Slice(None), 2}) -
                      boxes2.index({Slice(None), 0}) + offset) *
                     (boxes2.index({Slice(None), 3}) -
                      boxes2.index({Slice(None), 1}) + offset);
      ious = overlap / (area1.index({Slice(None), None}) + area2 - overlap);
    } else {
      ious = overlap / area1.index({Slice(None), None});
    }

    // #pragma omp parallel for

    // for (int i = 0; i < num_boxes1; i++) {
    //   for (int j = 0; j < num_boxes2; j++) {
    //     ious[i][j] =
    //         single_box_iou<T>(boxes1[i].data_ptr<T>(),
    //         boxes2[j].data_ptr<T>(),
    //                           offset, mode_flag);
    //   }
    // }
  }
}

void bbox_overlaps_cpu(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                       const int mode, const bool aligned, const int offset) {
  bbox_overlaps_cpu_kernel<float>(boxes1, boxes2, ious, mode, aligned, offset);
}

void bbox_overlaps_impl(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                        const int mode, const bool aligned, const int offset);

REGISTER_DEVICE_IMPL(bbox_overlaps_impl, CPU, bbox_overlaps_cpu);
