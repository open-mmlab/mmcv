#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void roi_pool_forward_npu(Tensor input, Tensor rois, Tensor output,
                          Tensor argmax, int pooled_height, int pooled_width,
                          float spatial_scale) {
  int64_t pooled_height_64 = pooled_height;
  int64_t pooled_width_64 = pooled_width;
  int64_t pooled_channel = 1;
  at::Tensor roi_actual_num =
      at::empty_like(rois, rois.options().dtype(at::kInt));
  if (input.sizes()[1] % 16 == 0) {
    OpCommand cmd;
    cmd.Name("RoiPoolingWithArgMax")
        .Input(input)
        .Input(rois)
        .Input(roi_actual_num)
        .Output(output)
        .Output(argmax)
        .Attr("pooled_h", pooled_height_64)
        .Attr("pooled_w", pooled_width_64)
        .Attr("spatial_scale_h", spatial_scale)
        .Attr("spatial_scale_w", spatial_scale)
        .Attr("pool_channel", pooled_channel)
        .Run();

  } else {
    OpCommand cmd;
    cmd.Name("RoiPoolingWithArgMax")
        .Input(input)
        .Input(rois)
        .Input(roi_actual_num)
        .Output(output)
        .Output(argmax)
        .Attr("pooled_h", pooled_height_64)
        .Attr("pooled_w", pooled_width_64)
        .Attr("spatial_scale_h", spatial_scale)
        .Attr("spatial_scale_w", spatial_scale)
        .Attr("pool_channel", pooled_channel)
        .Attr("_exclude_engines", (string) "AiCore")
        .Run();
  }
}

void roi_pool_backward_npu(Tensor grad_output, Tensor rois, Tensor argmax,
                           Tensor grad_input, int pooled_height,
                           int pooled_width, float spatial_scale) {
  int64_t pooled_height_64 = pooled_height;
  int64_t pooled_width_64 = pooled_width;
  int64_t pooled_channel = 1;
  at::Tensor argmax_trans = argmax.transpose(1, 2).transpose(2, 3);
  at::Tensor grad_output_trans = grad_output.transpose(1, 2).transpose(2, 3);
  at::Tensor roi_actual_num =
      at::empty_like(rois, rois.options().dtype(at::kInt));
  at::Tensor x = at::ones_like(grad_input).transpose(1, 2).transpose(2, 3);
  at::Tensor y = at::zeros_like(x);
  OpCommand cmd;
  cmd.Name("RoiPoolingGradWithArgMax")
      .Input(grad_output_trans)
      .Input(x)
      .Input(rois)
      .Input(roi_actual_num)
      .Input(argmax_trans)
      .Output(y)
      .Attr("pooled_h", pooled_height_64)
      .Attr("pooled_w", pooled_width_64)
      .Attr("spatial_scale_h", spatial_scale)
      .Attr("spatial_scale_w", spatial_scale)
      .Attr("pool_channel", pooled_channel)
      .Run();
  at::Tensor result = y.transpose(2, 3).transpose(1, 2);
  at::Tensor res = NpuUtils::format_contiguous(result);
  grad_input.copy_(res);
}

void roi_pool_forward_impl(Tensor input, Tensor rois, Tensor output,
                           Tensor argmax, int pooled_height, int pooled_width,
                           float spatial_scale);

void roi_pool_backward_impl(Tensor grad_output, Tensor rois, Tensor argmax,
                            Tensor grad_input, int pooled_height,
                            int pooled_width, float spatial_scale);

REGISTER_NPU_IMPL(roi_pool_forward_impl, roi_pool_forward_npu);
REGISTER_NPU_IMPL(roi_pool_backward_impl, roi_pool_backward_npu);
