#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void roi_align_forward_npu(Tensor input, Tensor rois, Tensor output,
                           Tensor argmax_y, Tensor argmax_x, int aligned_height,
                           int aligned_width, float spatial_scale,
                           int sampling_ratio, int pool_mode, bool aligned) {
  if (!aligned) {
    LOG(WARNING) << "The [aligned] attr in roi_align op is false";
  }
  int64_t aligned_height_64 = aligned_height;
  int64_t aligned_width_64 = aligned_width;
  int64_t sampling_ratio_64 = sampling_ratio;
  int64_t roi_end_mode = 0;
  OpCommand cmd;
  cmd.Name("ROIAlign")
      .Input(input)
      .Input(rois)
      .Output(output)
      .Attr("spatial_scale", spatial_scale)
      .Attr("pooled_height", aligned_height_64)
      .Attr("pooled_width", aligned_width_64)
      .Attr("sample_num", sampling_ratio_64)
      .Attr("roi_end_mode", roi_end_mode)
      .Run();
}

void roi_align_backward_npu(Tensor grad_output, Tensor rois, Tensor argmax_y,
                            Tensor argmax_x, Tensor grad_input,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned) {
  int64_t aligned_height_64 = aligned_height;
  int64_t aligned_width_64 = aligned_width;
  int64_t sampling_ratio_64 = sampling_ratio;
  int64_t roi_end_mode = 0;
  c10::SmallVector<int64_t, SIZE> xdiff_shape =
      array_to_small_vector(grad_input.sizes());
  OpCommand cmd;
  cmd.Name("ROIAlignGrad")
      .Input(grad_output)
      .Input(rois)
      .Output(grad_input)
      .Attr("xdiff_shape", xdiff_shape)
      .Attr("pooled_width", aligned_width_64)
      .Attr("pooled_height", aligned_height_64)
      .Attr("spatial_scale", spatial_scale)
      .Attr("sample_num", sampling_ratio_64)
      .Attr("roi_end_mode", roi_end_mode)
      .Run();
}

void roi_align_forward_impl(Tensor input, Tensor rois, Tensor output,
                            Tensor argmax_y, Tensor argmax_x,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned);

void roi_align_backward_impl(Tensor grad_output, Tensor rois, Tensor argmax_y,
                             Tensor argmax_x, Tensor grad_input,
                             int aligned_height, int aligned_width,
                             float spatial_scale, int sampling_ratio,
                             int pool_mode, bool aligned);

REGISTER_NPU_IMPL(roi_align_forward_impl, roi_align_forward_npu);
REGISTER_NPU_IMPL(roi_align_backward_impl, roi_align_backward_npu);
