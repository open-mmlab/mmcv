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
  bboxes1 = aligned ? bboxes1.transpose(0, 1) : bboxes1;
  bboxes2 = aligned ? bboxes2.transpose(0, 1) : bboxes2;
  OpCommand cmd;
  cmd.Name("Iou")
      .Input(bboxes2)
      .Input(bboxes1)
      .Output(ious)
      .Attr("mode", modeStr)
      .Attr("aligned", aligned)
      .Attr("eps", (float)offset)
      .Run();
}

REGISTER_NPU_IMPL(bbox_overlaps_impl, bbox_overlaps_npu);
