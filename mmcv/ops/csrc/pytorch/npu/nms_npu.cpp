#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

Tensor nms_npu(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
  TORCH_CHECK((boxes.scalar_type() == at::ScalarType::Float),
              "The type of boxes tensor passed in nms_npu should be float");
  int64_t offset_64 = offset;
  at::Tensor iou_threshold_y =
      at::empty({}, boxes.options().dtype(at::kFloat)).fill_(iou_threshold);
  at::Tensor scores_threshold_y =
      at::empty({}, boxes.options().dtype(at::kFloat)).fill_(0);
  at::Tensor max_outputsize_y =
      at::empty({}, boxes.options().dtype(at::kInt)).fill_(boxes.size(0));
  c10::SmallVector<int64_t, SIZE> outputsize = {boxes.size(0)};
  at::Tensor output =
      at::empty(outputsize, boxes.options().dtype(at::kInt)).fill_(-1);
  OpCommand cmd;
  cmd.Name("NonMaxSuppressionV3")
      .Input(boxes)
      .Input(scores)
      .Input(max_outputsize_y)
      .Input(iou_threshold_y)
      .Input(scores_threshold_y)
      .Attr("offset", offset_64)
      .Output(output)
      .Run();
  auto outputsizeBool = at::gt(output, -1);
  auto outputsizeInt = outputsizeBool.to(at::kInt);
  auto countLen = at::sum(outputsizeInt, at::kInt);
  at::Tensor actual_output = output.slice(0, 0, countLen.item().toLong());
  actual_output = actual_output.to(at::kLong);
  return actual_output;
}

Tensor nms_impl(Tensor boxes, Tensor scores, float iou_threshold, int offset);

REGISTER_NPU_IMPL(nms_impl, nms_npu);
