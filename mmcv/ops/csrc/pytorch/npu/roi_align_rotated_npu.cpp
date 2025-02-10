#include "common_util.h"
#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void roi_align_rotated_forward_npu(Tensor input, Tensor rois, Tensor output,
                                   int aligned_height, int aligned_width,
                                   float spatial_scale, int sampling_ratio,
                                   bool aligned, bool clockwise) {
  int64_t aligned_height_64 = aligned_height;
  int64_t aligned_width_64 = aligned_width;
  int64_t sampling_ratio_64 = sampling_ratio;

  at::Tensor input_trans = input.permute({0, 2, 3, 1}).contiguous();
  at::Tensor rois_trans = rois.permute({1, 0}).contiguous();
  at::Tensor output_trans = output.permute({0, 2, 3, 1}).contiguous();

  OpCommand cmd;
  cmd.Name("RoiAlignRotated")
      .Input(input_trans)
      .Input(rois_trans)
      .Output(output_trans)
      .Attr("pooled_h", aligned_height_64)
      .Attr("pooled_w", aligned_width_64)
      .Attr("spatial_scale", spatial_scale)
      .Attr("sampling_ratio", sampling_ratio_64)
      .Attr("aligned", aligned)
      .Attr("clockwise", clockwise)
      .Run();

  output_trans = output_trans.permute({0, 3, 1, 2}).contiguous();
  output.copy_(output_trans);
}

void roi_align_rotated_backward_npu(Tensor top_grad, Tensor rois,
                                    Tensor bottom_grad, int aligned_height,
                                    int aligned_width, float spatial_scale,
                                    int sampling_ratio, bool aligned,
                                    bool clockwise) {
  int64_t aligned_height_64 = aligned_height;
  int64_t aligned_width_64 = aligned_width;
  int64_t sampling_ratio_64 = sampling_ratio;

  at::Tensor top_grad_trans = top_grad.permute({0, 2, 3, 1}).contiguous();
  at::Tensor rois_trans = rois.permute({1, 0}).contiguous();
  at::Tensor bottom_grad_trans = bottom_grad.permute({0, 2, 3, 1}).contiguous();

  c10::SmallVector<int64_t, 8> y_grad_shape;
  auto shape = bottom_grad_trans.sizes();
  for (uint64_t i = 0; i < shape.size(); i++) {
    y_grad_shape.emplace_back(shape[i]);
  }
  OpCommand cmd;
  cmd.Name("RoiAlignRotatedGrad")
      .Input(top_grad_trans)
      .Input(rois_trans)
      .Output(bottom_grad_trans)
      .Attr("y_grad_shape", y_grad_shape)
      .Attr("pooled_h", aligned_width_64)
      .Attr("pooled_w", aligned_height_64)
      .Attr("spatial_scale", spatial_scale)
      .Attr("sampling_ratio", sampling_ratio_64)
      .Attr("aligned", aligned)
      .Attr("clockwise", clockwise)
      .Run();

  bottom_grad_trans = bottom_grad_trans.permute({0, 3, 1, 2}).contiguous();
  bottom_grad.copy_(bottom_grad_trans);
}

void roi_align_rotated_forward_impl(Tensor input, Tensor rois, Tensor output,
                                    int aligned_height, int aligned_width,
                                    float spatial_scale, int sampling_ratio,
                                    bool aligned, bool clockwise);

void roi_align_rotated_backward_impl(Tensor top_grad, Tensor rois,
                                     Tensor bottom_grad, int aligned_height,
                                     int aligned_width, float spatial_scale,
                                     int sampling_ratio, bool aligned,
                                     bool clockwise);

REGISTER_NPU_IMPL(roi_align_rotated_forward_impl,
                  roi_align_rotated_forward_npu);
REGISTER_NPU_IMPL(roi_align_rotated_backward_impl,
                  roi_align_rotated_backward_npu);
