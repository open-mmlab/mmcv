#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void iou3d_boxes_overlap_bev_forward_impl(const int num_a, const Tensor boxes_a,
                                          const int num_b, const Tensor boxes_b,
                                          Tensor ans_overlap);

void iou3d_boxes_overlap_bev_forward_npu(const int num_a, const Tensor boxes_a,
                                         const int num_b, const Tensor boxes_b,
                                         Tensor ans_overlap) {
  TORCH_CHECK(boxes_a.size(1) == 7, "boxes_a must be 2D tensor (N, 7)");
  TORCH_CHECK(boxes_b.size(1) == 7, "boxes_b must be 2D tensor (N, 7)");

  auto format_flag = 3;
  auto clockwise = true;
  auto mode_flag = 0;
  auto aligned = false;
  auto margin = 1e-2;

  EXEC_NPU_CMD(aclnnBoxesOverlapBev, boxes_a, boxes_b, format_flag, clockwise,
               mode_flag, aligned, margin, ans_overlap);
  return;
}

REGISTER_NPU_IMPL(iou3d_boxes_overlap_bev_forward_impl,
                  iou3d_boxes_overlap_bev_forward_npu);
