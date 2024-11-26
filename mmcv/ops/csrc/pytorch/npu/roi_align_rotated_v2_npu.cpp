#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void roi_align_rotated_v2_forward_npu(const Tensor x, Tensor rois_map,
                                    Tensor y,
                                    int32_t pooled_h,
                                    int32_t pooled_w,
                                    double spatial_scale,
                                    int32_t sampling_ratio,
                                    bool aligned,
                                    bool clockwise) {
  at::Tensor feature_map = x.permute({0, 2, 3, 1}).contiguous();
  at::Tensor rois = rois_map.permute({1, 0}).contiguous();
  at_npu::native::OpCommand cmd;
  cmd.Name("RoiAlignRotated")
      .Input(feature_map)
      .Input(rois)
      .Output(y)
      .Attr("pooled_h", static_cast<int64_t>(pooled_h))
      .Attr("pooled_w", static_cast<int64_t>(pooled_w))
      .Attr("spatial_scale", static_cast<float>(spatial_scale))
      .Attr("sampling_ratio", static_cast<int64_t>(sampling_ratio))
      .Attr("aligned", aligned)
      .Attr("clockwise", clockwise)
      .Run();
}

void roi_align_rotated_v2_forward_impl(const Tensor x, Tensor rois,
                                    Tensor y,
                                    int32_t pooled_h,
                                    int32_t pooled_w,
                                    double spatial_scale,
                                    int32_t sampling_ratio,
                                    bool aligned,
                                    bool clockwise);

REGISTER_NPU_IMPL(roi_align_rotated_v2_forward_impl, roi_align_rotated_v2_forward_npu);

void roi_align_rotated_v2_backward_npu(const Tensor input, Tensor rois,
                                    Tensor grad_output, Tensor grad_input,
                                    int32_t pooled_height,
                                    int32_t pooled_width,
                                    double spatial_scale,
                                    int32_t sampling_ratio,
                                    bool aligned,
                                    bool clockwise) {
  EXEC_NPU_CMD(aclnnRoiAlignRotatedGradV2, input, rois, grad_output,
               pooled_height, pooled_width, spatial_scale, sampling_ratio, aligned, clockwise,
               grad_input);
}

void roi_align_rotated_v2_backward_impl(const Tensor input, Tensor rois,
                                    Tensor grad_output, Tensor grad_input,
                                    int32_t pooled_height,
                                    int32_t pooled_width,
                                    double spatial_scale,
                                    int32_t sampling_ratio,
                                    bool aligned,
                                    bool clockwise);

REGISTER_NPU_IMPL(roi_align_rotated_v2_backward_impl, roi_align_rotated_v2_backward_npu);
