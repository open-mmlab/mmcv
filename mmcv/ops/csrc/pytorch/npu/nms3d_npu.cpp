#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

constexpr int32_t BOX_DIM = 7;

void iou3d_nms3d_forward_npu(const Tensor boxes, Tensor &keep, Tensor &num_out,
                             float nms_overlap_thresh) {
  TORCH_CHECK((boxes.sizes()[1] == BOX_DIM),
              "Input boxes shape should be (N, 7)");
  int32_t box_num = boxes.size(0);
  int32_t data_align = 16;
  int32_t mask_num = ((box_num - 1) / data_align + 1) * data_align;
  const double iou_threshold = nms_overlap_thresh;
  at::Tensor mask =
      at::empty({box_num, mask_num}, boxes.options().dtype(at::kShort));
  EXEC_NPU_CMD(aclnnNms3d, boxes, iou_threshold, mask);

  Tensor keep_t = at::zeros({box_num}, mask.options());
  Tensor num_out_t = at::zeros(1, mask.options());
  EXEC_NPU_CMD(aclnnGatherNms3dMask, mask, keep_t, num_out_t);
  num_out.fill_(num_out_t.item().toLong());
  keep.copy_(keep_t);
}

void iou3d_nms3d_forward_impl(const Tensor boxes, Tensor &keep, Tensor &num_out,
                              float nms_overlap_thresh);

REGISTER_NPU_IMPL(iou3d_nms3d_forward_impl, iou3d_nms3d_forward_npu);
