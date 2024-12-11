#include "pytorch_npu_helper.hpp"
using namespace NPU_NAME_SPACE;
using namespace std;

void roiaware_pool3d_forward_npu(int boxes_num, int pts_num, int channels,
                                 int max_pts_each_voxel, int out_x, int out_y,
                                 int out_z, const Tensor rois, const Tensor pts,
                                 const Tensor pts_feature, Tensor argmax,
                                 Tensor pts_idx_of_voxels,
                                 Tensor pooled_features, int pool_method) {
  at::Tensor rois_cast = rois;
  at::Tensor pts_cast = pts;
  at::Tensor pts_feature_cast = pts_feature;
  at::Tensor pooled_features_cast = pooled_features;

  auto dtype = rois.dtype();
  if (dtype == at::kHalf) {
    rois_cast = rois_cast.to(at::kFloat);
    pts_cast = pts_cast.to(at::kFloat);
    pts_feature_cast = pts_feature_cast.to(at::kFloat);
    pooled_features_cast = pooled_features_cast.to(at::kFloat);
  }

  EXEC_NPU_CMD(aclnnRoiawarePool3d, rois_cast, pts_cast, pts_feature_cast,
               pool_method, max_pts_each_voxel, out_x, out_y, out_z, argmax,
               pts_idx_of_voxels, pooled_features_cast);

  if (dtype == at::kHalf) {
    pooled_features_cast = pooled_features_cast.to(at::kHalf);
  }

  pooled_features.copy_(pooled_features_cast);
}

void roiaware_pool3d_backward_npu(int boxes_num, int out_x, int out_y,
                                  int out_z, int channels,
                                  int max_pts_each_voxel,
                                  const Tensor pts_idx_of_voxels,
                                  const Tensor argmax, const Tensor grad_out,
                                  Tensor grad_in, int pool_method) {
  int32_t npoints = grad_in.size(0);

  auto dtype = grad_out.dtype();
  at::Tensor grad_out_cast = grad_out;
  at::Tensor grad_in_cast = grad_in;

  if (dtype == at::kHalf) {
    grad_out_cast = grad_out.to(at::kFloat);
    grad_in_cast = grad_in_cast.to(at::kFloat);
  }

  if (pool_method == 0) {
    // maxpool3d
    EXEC_NPU_CMD(aclnnRoiawareMaxpool3dGrad, argmax, grad_out_cast, boxes_num,
                 out_x, out_y, out_z, channels, npoints, grad_in_cast);
  } else if (pool_method == 1) {
    // avgpool3d
    EXEC_NPU_CMD(aclnnRoiawareAvgpool3dGrad, pts_idx_of_voxels, grad_out_cast,
                 boxes_num, out_x, out_y, out_z, channels, npoints,
                 max_pts_each_voxel, grad_in_cast);
  }

  if (dtype == at::kHalf) {
    grad_in_cast = grad_in_cast.to(at::kHalf);
  }

  grad_in.copy_(grad_in_cast);
}

void roiaware_pool3d_forward_impl(int boxes_num, int pts_num, int channels,
                                  int max_pts_each_voxel, int out_x, int out_y,
                                  int out_z, const Tensor rois,
                                  const Tensor pts, const Tensor pts_feature,
                                  Tensor argmax, Tensor pts_idx_of_voxels,
                                  Tensor pooled_features, int pool_method);

void roiaware_pool3d_backward_impl(int boxes_num, int out_x, int out_y,
                                   int out_z, int channels,
                                   int max_pts_each_voxel,
                                   const Tensor pts_idx_of_voxels,
                                   const Tensor argmax, const Tensor grad_out,
                                   Tensor grad_in, int pool_method);

REGISTER_NPU_IMPL(roiaware_pool3d_forward_impl, roiaware_pool3d_forward_npu);
REGISTER_NPU_IMPL(roiaware_pool3d_backward_impl, roiaware_pool3d_backward_npu);
