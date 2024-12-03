#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void border_align_forward_impl(const Tensor &input, const Tensor &boxes,
                               Tensor output, Tensor argmax_idx,
                               const int pool_size);

void border_align_forward_npu(const Tensor &input, const Tensor &boxes,
                              Tensor output, Tensor argmax_idx,
                              const int pool_size) {
  TORCH_CHECK(input.size(0) == boxes.size(0),
              "The batch sizes of feature map and rois must be the same.");
  TORCH_CHECK(input.size(1) % 4 == 0,
              "The number of channels must be divisible by 4.");
  TORCH_CHECK(pool_size >= 2, "The pool size should be larger than 2.");
  int32_t batch_size = input.size(0);
  int32_t channels = input.size(1);
  int32_t height = input.size(2);
  int32_t width = input.size(3);
  at::Tensor feature_map = input.permute({0, 2, 3, 1}).contiguous();
  at::Tensor rois_map = boxes.contiguous();
  at::Tensor temp_tensor = at::zeros(
      {batch_size, height * width, pool_size + 1, channels}, input.options());
  EXEC_NPU_CMD(aclnnBorderAlign, feature_map, rois_map, pool_size, temp_tensor);
  auto max_result = temp_tensor.max(-2);
  at::Tensor output_ = std::get<0>(max_result).to(at::kFloat);
  output_ = output_.reshape({batch_size, height * width, 4, channels / 4})
                .permute({0, 3, 1, 2})
                .contiguous();
  output.copy_(output_);
  at::Tensor argmax_idx_ = std::get<1>(max_result).to(at::kInt);
  argmax_idx_ =
      argmax_idx_.reshape({batch_size, height * width, 4, channels / 4})
          .permute({0, 3, 1, 2})
          .contiguous();
  argmax_idx.copy_(argmax_idx_);
}
REGISTER_NPU_IMPL(border_align_forward_impl, border_align_forward_npu);

void border_align_backward_impl(const Tensor &grad_output, const Tensor &boxes,
                                const Tensor &argmax_idx, Tensor grad_input,
                                const int pool_size);

void border_align_backward_npu(const Tensor &grad_output, const Tensor &boxes,
                               const Tensor &argmax_idx, Tensor grad_input,
                               const int pool_size) {
  TORCH_CHECK(grad_output.dim() == 4,
              "grad_out.dim() must be 4, but got: ", grad_output.dim());
  TORCH_CHECK(boxes.dim() == 3, "idx.dim() must be 3, but got: ", boxes.dim());
  TORCH_CHECK(argmax_idx.dim() == 4,
              "argmax_idx.dim() must be 4, but got: ", argmax_idx.dim());

  int32_t batch_size = grad_output.size(0);
  int32_t feat_channels = grad_output.size(1) * 4;
  int32_t channels = grad_output.size(1);
  int32_t box_size = boxes.size(1);
  int32_t height = grad_input.size(2);
  int32_t width = grad_input.size(3);

  EXEC_NPU_CMD(aclnnBorderAlignGrad, grad_output, boxes, argmax_idx, channels,
               box_size, height, width, pool_size, batch_size, grad_input);
}
REGISTER_NPU_IMPL(border_align_backward_impl, border_align_backward_npu);
