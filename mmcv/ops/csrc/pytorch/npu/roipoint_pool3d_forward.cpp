#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void roipoint_pool3d_forward_impl_npu(int batch_size, int pts_num,
                                      int boxes_num, int feature_in_len,
                                      int sampled_pts_num, const Tensor xyz,
                                      const Tensor boxes3d,
                                      const Tensor pts_feature,
                                      Tensor pooled_features,
                                      Tensor pooled_empty_flag) {
  auto points_trans = xyz.transpose(1, 2).contiguous();
  auto point_features_trans = pts_feature.transpose(1, 2).contiguous();
  c10::SmallVector<int64_t, 8> features_trans_size = {
      xyz.size(0), boxes3d.size(1), xyz.size(2) + pts_feature.size(2),
      sampled_pts_num};
  at::Tensor pooled_features_trans =
      at::empty(features_trans_size, xyz.options());
  c10::SmallVector<int64_t, 8> empty_flag_size = {boxes3d.size(0),
                                                     boxes3d.size(1)};
  EXEC_NPU_CMD(aclnnRoipointPool3dForward, points_trans, point_features_trans,
               boxes3d, sampled_pts_num, pooled_features_trans,
               pooled_empty_flag);
  auto pooled_features_cache =
      pooled_features_trans.transpose(2, 3).contiguous();
  pooled_features.copy_(pooled_features_cache);
}

void roipoint_pool3d_forward_impl(int batch_size, int pts_num, int boxes_num,
                                  int feature_in_len, int sampled_pts_num,
                                  const Tensor xyz, const Tensor boxes3d,
                                  const Tensor pts_feature,
                                  Tensor pooled_features,
                                  Tensor pooled_empty_flag);

REGISTER_NPU_IMPL(roipoint_pool3d_forward_impl,
                  roipoint_pool3d_forward_impl_npu);
