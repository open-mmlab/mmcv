#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;

Tensor nms_rotated_npu(const Tensor dets, const Tensor scores,
                       const Tensor labels, const float iou_threshold) {
  auto originDtype = dets.scalar_type();
  at::Tensor detsCast = dets;
  at::Tensor scoresCast = scores;
  if (originDtype != at::kFloat) {
    detsCast = detsCast.to(at::kFloat);
    scoresCast = scoresCast.to(at::kFloat);
  }
  c10::SmallVector<int64_t, SIZE> selectedIndexSize = {dets.size(0)};
  at::Tensor selectedBox = at::empty_like(dets);
  at::Tensor selectedIndex =
      at::empty(selectedIndexSize, dets.options().dtype(at::kInt));

  c10::SmallVector<int64_t, N> output_sync_idx = {0, 1};
  OpCommand cmd;
  cmd.Sync(output_sync_idx)
      .Name("RotatedNMS")
      .Input(detsCast)
      .Input(scoresCast)
      .Input(labels)
      .Output(selectedBox)
      .Output(selectedIndex)
      .Attr("iou_threshold", (float)iou_threshold)
      .Run();
  selectedIndex = selectedIndex.to(at::kLong);
  return selectedIndex;
}
