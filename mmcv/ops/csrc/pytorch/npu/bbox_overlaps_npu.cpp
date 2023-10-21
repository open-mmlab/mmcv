#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void bbox_overlaps_impl(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                        const int mode, const bool aligned, const int offset);

void bbox_overlaps_npu(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                       const int mode, const bool aligned, const int offset) {
  string modeStr = "iou";
  if (mode == 1) {
    modeStr = "iof";
  }
  bool swap_flag = false;
  at::Tensor bboxesFP32 = bboxes2;
  at::Tensor gtboxesFP32 = bboxes1;
  if (bboxes2.size(0) < bboxes1.size(0)) {
    swap_flag = true;
    bboxesFP32 = bboxes1;
    gtboxesFP32 = bboxes2;
  }
  if (bboxes2.scalar_type() != at::kFloat) {
    bboxesFP32 = bboxesFP32.to(at::kFloat);
    gtboxesFP32 = gtboxesFP32.to(at::kFloat);
  }
  c10::SmallVector<int64_t, SIZE> iousSize = {gtboxesFP32.size(0),
                                              bboxesFP32.size(0)};
  if (aligned) {
    iousSize = {gtboxesFP32.size(0), 1};
  }
  at::Tensor iousFP32 = at::empty(iousSize, bboxesFP32.options());
  bboxesFP32 = aligned ? bboxesFP32.transpose(0, 1) : bboxesFP32;
  gtboxesFP32 = aligned ? gtboxesFP32.transpose(0, 1) : gtboxesFP32;
  OpCommand cmd;
  cmd.Name("Iou")
      .Input(bboxesFP32)
      .Input(gtboxesFP32)
      .Output(iousFP32)
      .Attr("mode", modeStr)
      .Attr("eps", (float)offset)
      .Attr("aligned", aligned)
      .Run();
  if (bboxes2.scalar_type() != at::kFloat) {
    iousFP32 = iousFP32.to(at::kHalf);
  }
  iousFP32 = swap_flag ? iousFP32.transpose(0, 1) : iousFP32;
  ious.copy_(iousFP32);
}

REGISTER_NPU_IMPL(bbox_overlaps_impl, bbox_overlaps_npu);
