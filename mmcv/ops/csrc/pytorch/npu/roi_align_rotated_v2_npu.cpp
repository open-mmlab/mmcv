#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void roi_align_rotated_v2_forward_npu(const Tensor input, Tensor rois_map,
                                    Tensor output,
                                    double spatial_scale,
                                    int32_t sampling_ratio,
                                    int32_t pooled_height,
                                    int32_t pooled_width,
                                    bool aligned,
                                    bool clockwise) {
  at::Tensor feature_map = input.permute({0, 2, 3, 1}).contiguous();
  at::Tensor rois = rois_map.permute({1, 0}).contiguous();
  EXEC_NPU_CMD(aclnnRoiAlignRotatedV2, feature_map, rois, spatial_scale, sampling_ratio, pooled_height, pooled_width, aligned, clockwise, output);
}

void roi_align_rotated_v2_forward_impl(const Tensor input, Tensor rois,
                                    Tensor output,
                                    double spatial_scale,
                                    int32_t sampling_ratio,
                                    int32_t pooled_height,
                                    int32_t pooled_width,
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
