#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void rotated_feature_align_forward_impl(const Tensor features,
                                        const Tensor best_bboxes,
                                        const float spatial_scale,
                                        const int points, Tensor output);

void rotated_feature_align_backward_impl(const Tensor top_grad,
                                         const Tensor best_bboxes,
                                         const float spatial_scale,
                                         const int points, Tensor bottom_grad);

void rotated_feature_align_forward_npu(const Tensor features,
                                       const Tensor best_bboxes,
                                       const float spatial_scale,
                                       const int points, Tensor output) {
  int64_t points_ = (int64_t)points;
  at::Tensor best_bboxes_ = best_bboxes.transpose(2, 3).transpose(1, 2);
  OpCommand cmd;
  cmd.Name("RotatedFeatureAlign")
      .Input(features)
      .Input(best_bboxes_)
      .Output(output)
      .Attr("spatial_scale", spatial_scale)
      .Attr("points", points_)
      .Run();
}

void rotated_feature_align_backward_npu(const Tensor top_grad,
                                        const Tensor best_bboxes,
                                        const float spatial_scale,
                                        const int points, Tensor bottom_grad) {
  int64_t points_ = (int64_t)points;
  at::Tensor best_bboxes_ = best_bboxes.transpose(2, 3).transpose(1, 2);
  OpCommand cmd;
  cmd.Name("RotatedFeatureAlignGrad")
      .Input(top_grad)
      .Input(best_bboxes_)
      .Output(bottom_grad)
      .Attr("spatial_scale", spatial_scale)
      .Attr("points", points_)
      .Run();
}

REGISTER_NPU_IMPL(rotated_feature_align_forward_impl,
                  rotated_feature_align_forward_npu);

REGISTER_NPU_IMPL(rotated_feature_align_backward_impl,
                  rotated_feature_align_backward_npu);
