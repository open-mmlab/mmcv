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
  at::Tensor bboxesFP32 = bboxes2;
  at::Tensor gtboxesFP32 = bboxes1;
  if (bboxes2.scalar_type() != at::ScalarType::Float) {
    bboxesFP32 = NPUNativeFunctions::npu_dtype_cast(bboxes2, at::kFloat);
    gtboxesFP32 = NPUNativeFunctions::npu_dtype_cast(bboxes1, at::kFloat);
  }
  c10::SmallVector<int64_t, SIZE> iousSize = {gtboxesFP32.size(0),
                                              bboxesFP32.size(0)};
  if (aligned) {
    iousSize = {gtboxesFP32.size(0), 1};
  }
  at::Tensor iousFP32 = OpPreparation::ApplyTensor(bboxesFP32, iousSize);
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
  if (bboxes2.scalar_type() != at::ScalarType::Float) {
    iousFP32 = NPUNativeFunctions::npu_dtype_cast(iousFP32, at::kHalf);
  }
  ious.copy_(iousFP32);
}

REGISTER_NPU_IMPL(bbox_overlaps_impl, bbox_overlaps_npu);
