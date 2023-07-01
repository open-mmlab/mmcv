#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;

void rotated_feature_align_backward_impl(const Tensor top_grad,
                                         const Tensor best_bboxes,
                                         const float spatial_scale,
                                         const int points, Tensor bottom_grad);

void rotated_feature_align_backward_npu(const Tensor top_grad,
                                        const Tensor best_bboxes,
                                        const float spatial_scale,
                                        const int points, Tensor bottom_grad) {
  int64_t points_ = (int64_t)points;
  OpCommand cmd;
  cmd.Name("RotatedFeatureAlignGrad")
      .Input(top_grad)
      .Input(best_bboxes)
      .Output(bottom_grad)
      .Attr("spatial_scale", spatial_scale)
      .Attr("points", points_)
      .Run();
}

REGISTER_NPU_IMPL(rotated_feature_align_backward_impl,
                  rotated_feature_align_backward_npu);
