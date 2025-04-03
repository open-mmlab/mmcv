#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void box_iou_rotated_impl(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                          const int mode_flag, const bool aligned);

void box_iou_rotated_npu(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                         const int mode_flag, const bool aligned) {
  TORCH_CHECK(boxes1.size(1) == 5, "boxes1 must be 2D tensor (N, 5)");
  TORCH_CHECK(boxes1.size(1) == 5, "boxes1 must be 2D tensor (N, 5)");
  EXEC_NPU_CMD(aclnnBoxIou, boxes1, boxes2, mode_flag, aligned, ious);
  return;
}

REGISTER_NPU_IMPL(box_iou_rotated_impl, box_iou_rotated_npu);
