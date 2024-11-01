#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void box_iou_quadri_impl(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                         const int mode_flag, const bool aligned);

void box_iou_quadri_npu(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                        const int mode_flag, const bool aligned) {
  TORCH_CHECK(boxes1.size(1) == 8, "boxes1 must be 2D tensor (N, 8)");
  TORCH_CHECK(boxes1.size(1) == 8, "boxes1 must be 2D tensor (N, 8)");

  EXEC_NPU_CMD(aclnnBoxIou, boxes1, boxes2, mode_flag, aligned, ious);
  return;
}

REGISTER_NPU_IMPL(box_iou_quadri_impl, box_iou_quadri_npu);
