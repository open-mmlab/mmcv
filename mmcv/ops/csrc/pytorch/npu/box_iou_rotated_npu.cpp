#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void box_iou_rotated_impl(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                          const int mode_flag, const bool aligned);

void box_iou_rotated_npu(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                         const int mode_flag, const bool aligned) {
  at::Tensor boxes = at::ones_like(boxes1);
  at::Tensor query_boxes = at::ones_like(boxes2);
  boxes = boxes1.transpose(0, 1).unsqueeze(0);
  query_boxes = boxes2.transpose(0, 1).unsqueeze(0);

  bool is_trans = false;
  string modeStr = "iou";
  if (mode_flag == 1) {
    modeStr = "iof";
  }
  bool is_cross = true;
  if (aligned) {
    is_cross = false;
  }
  float v_threshold = 0;
  float e_threshold = 0;

  OpCommand cmd;
  cmd.Name("RotatedIou")
      .Input(boxes)
      .Input(query_boxes)
      .Output(ious)
      .Attr("trans", is_trans)
      .Attr("mode", modeStr)
      .Attr("is_cross", is_cross)
      .Attr("v_threshold", v_threshold)
      .Attr("e_threshold", e_threshold)
      .Run();

  if (is_cross) {
    ious = ious.view({boxes1.size(0), boxes2.size(0)});
  } else {
    ious = ious.view({boxes1.size(0), 1});
  }
}

REGISTER_NPU_IMPL(box_iou_rotated_impl, box_iou_rotated_npu);
