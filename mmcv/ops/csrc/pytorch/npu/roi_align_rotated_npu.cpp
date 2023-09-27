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
  OpCommand cmd;
  cmd.Name("RoiAlignRotated")
      .Input(input)
      .Input(rois)
      .Output(output)
      .Attr("pooled_h", aligned_height_64)
      .Attr("pooled_w", aligned_width_64)
      .Attr("spatial_scale", spatial_scale)
      .Attr("sampling_ratio", sampling_ratio_64)
      .Attr("aligned", aligned)
      .Attr("clockwise", clockwise)
      .Run();
}

void roi_align_rotated_backward_npu(Tensor top_grad, Tensor rois,
                                    Tensor bottom_grad, int aligned_height,
                                    int aligned_width, float spatial_scale,
                                    int sampling_ratio, bool aligned,
                                    bool clockwise) {
  int64_t aligned_height_64 = aligned_height;
  int64_t aligned_width_64 = aligned_width;
  int64_t sampling_ratio_64 = sampling_ratio;
  c10::SmallVector<int64_t, SIZE> y_grad_shape =
      array_to_small_vector(bottom_grad.sizes());
  OpCommand cmd;
  cmd.Name("RoiAlignRotatedGrad")
      .Input(top_grad)
      .Input(rois)
      .Output(bottom_grad)
      .Attr("y_grad_shape", y_grad_shape)
      .Attr("pooled_h", aligned_width_64)
      .Attr("pooled_w", aligned_height_64)
      .Attr("spatial_scale", spatial_scale)
      .Attr("sampling_ratio", sampling_ratio_64)
      .Attr("aligned", aligned)
      .Attr("clockwise", clockwise)
      .Run();
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
