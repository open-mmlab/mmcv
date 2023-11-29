#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void points_in_boxes_part_forward_impl_npu(int batch_size, int boxes_num,
                                           int pts_num, const Tensor boxes,
                                           const Tensor pts,
                                           Tensor box_idx_of_points) {
  c10::SmallVector<int64_t, 8> output_size = {pts.size(0), pts.size(1)};
  auto boxes_trans = boxes.transpose(1, 2).contiguous();
  EXEC_NPU_CMD(aclnnPointsInBox, boxes_trans, pts, box_idx_of_points);
}
void points_in_boxes_part_forward_impl(int batch_size, int boxes_num,
                                       int pts_num, const Tensor boxes,
                                       const Tensor pts,
                                       Tensor box_idx_of_points);
REGISTER_NPU_IMPL(points_in_boxes_part_forward_impl,
                  points_in_boxes_part_forward_impl_npu);
