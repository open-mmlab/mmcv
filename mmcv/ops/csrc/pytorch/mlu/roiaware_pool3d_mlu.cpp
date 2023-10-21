/*************************************************************************
 * Copyright (C) 2022 by Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "mlu_common_helper.h"

void RoiawarePool3dForwardMLUKernelLauncher(
    const int pool_method, const int boxes_num, const int pts_num,
    const int channels, const int max_pts_each_voxel, const int out_x,
    const int out_y, const int out_z, const Tensor rois, const Tensor pts,
    const Tensor pts_feature, Tensor pts_idx_of_voxels, Tensor pooled_features,
    Tensor argmax) {
  // get compute handle
  auto handle = mluOpGetCurrentHandle();

  auto rois_contiguous =
      torch_mlu::cnnl::ops::cnnl_contiguous(rois, rois.suggest_memory_format());
  auto pts_contiguous =
      torch_mlu::cnnl::ops::cnnl_contiguous(pts, pts.suggest_memory_format());
  auto pts_feature_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      pts_feature, pts_feature.suggest_memory_format());
  auto argmax_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      argmax, argmax.suggest_memory_format());
  auto pts_idx_of_voxels_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      pts_idx_of_voxels, pts_idx_of_voxels.suggest_memory_format());
  auto pooled_features_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      pooled_features, pooled_features.suggest_memory_format());

  MluOpTensorDescriptor rois_desc, pts_desc, pts_feature_desc, argmax_desc,
      pts_idx_of_voxels_desc, pooled_features_desc;
  rois_desc.set(rois_contiguous);
  pts_desc.set(pts_contiguous);
  pts_feature_desc.set(pts_feature_contiguous);
  argmax_desc.set(argmax_contiguous);
  pts_idx_of_voxels_desc.set(pts_idx_of_voxels_contiguous);
  pooled_features_desc.set(pooled_features_contiguous);

  // allocate extra space for workspace
  size_t workspace_size = 0;
  TORCH_MLUOP_CHECK(mluOpGetRoiawarePool3dForwardWorkspaceSize(
      handle, rois_desc.desc(), pts_desc.desc(), pts_feature_desc.desc(),
      &workspace_size));

  auto workspace = at::empty(workspace_size, rois.options().dtype(at::kByte));
  auto workspace_impl = torch_mlu::getMluTensorImpl(workspace);
  auto workspace_ptr = workspace_impl->cnnlMalloc();

  auto rois_impl = torch_mlu::getMluTensorImpl(rois_contiguous);
  auto pts_impl = torch_mlu::getMluTensorImpl(pts_contiguous);
  auto pts_feature_impl = torch_mlu::getMluTensorImpl(pts_feature_contiguous);
  auto argmax_impl = torch_mlu::getMluTensorImpl(argmax_contiguous);
  auto pts_idx_of_voxels_impl =
      torch_mlu::getMluTensorImpl(pts_idx_of_voxels_contiguous);
  auto pooled_features_impl =
      torch_mlu::getMluTensorImpl(pooled_features_contiguous);

  auto rois_ptr = rois_impl->cnnlMalloc();
  auto pts_ptr = pts_impl->cnnlMalloc();
  auto pts_feature_ptr = pts_feature_impl->cnnlMalloc();
  auto argmax_ptr = argmax_impl->cnnlMalloc();
  auto pts_idx_of_voxels_ptr = pts_idx_of_voxels_impl->cnnlMalloc();
  auto pooled_features_ptr = pooled_features_impl->cnnlMalloc();

  CNLOG(INFO) << "Call mluOpRoiawarePool3dForward().";
  TORCH_MLUOP_CHECK(mluOpRoiawarePool3dForward(
      handle, pool_method, boxes_num, pts_num, channels, rois_desc.desc(),
      rois_ptr, pts_desc.desc(), pts_ptr, pts_feature_desc.desc(),
      pts_feature_ptr, workspace_ptr, workspace_size, max_pts_each_voxel, out_x,
      out_y, out_z, argmax_desc.desc(), argmax_ptr,
      pts_idx_of_voxels_desc.desc(), pts_idx_of_voxels_ptr,
      pooled_features_desc.desc(), pooled_features_ptr));
}

void roiaware_pool3d_forward_mlu(int boxes_num, int pts_num, int channels,
                                 int max_pts_each_voxel, int out_x, int out_y,
                                 int out_z, const Tensor rois, const Tensor pts,
                                 const Tensor pts_feature, Tensor argmax,
                                 Tensor pts_idx_of_voxels,
                                 Tensor pooled_features, int pool_method) {
  RoiawarePool3dForwardMLUKernelLauncher(
      pool_method, boxes_num, pts_num, channels, max_pts_each_voxel, out_x,
      out_y, out_z, rois, pts, pts_feature, pts_idx_of_voxels, pooled_features,
      argmax);
}

void roiaware_pool3d_forward_impl(int boxes_num, int pts_num, int channels,
                                  int max_pts_each_voxel, int out_x, int out_y,
                                  int out_z, const Tensor rois,
                                  const Tensor pts, const Tensor pts_feature,
                                  Tensor argmax, Tensor pts_idx_of_voxels,
                                  Tensor pooled_features, int pool_method);

REGISTER_DEVICE_IMPL(roiaware_pool3d_forward_impl, MLU,
                     roiaware_pool3d_forward_mlu);

void RoiawarePool3dBackwardMLUKernelLauncher(
    int pool_method, int boxes_num, int out_x, int out_y, int out_z,
    int channels, int max_pts_each_voxel, const Tensor pts_idx_of_voxels,
    const Tensor argmax, const Tensor grad_out, Tensor grad_in) {
  // get compute handle
  auto handle = mluOpGetCurrentHandle();
  auto pts_idx_of_voxels_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      pts_idx_of_voxels, pts_idx_of_voxels.suggest_memory_format());
  auto argmax_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      argmax, argmax.suggest_memory_format());
  auto grad_out_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      grad_out, grad_out.suggest_memory_format());
  auto grad_in_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      grad_in, grad_in.suggest_memory_format());

  MluOpTensorDescriptor pts_idx_of_voxels_desc, argmax_desc, grad_out_desc,
      grad_in_desc;

  pts_idx_of_voxels_desc.set(pts_idx_of_voxels_contiguous);
  argmax_desc.set(argmax_contiguous);
  grad_out_desc.set(grad_out_contiguous);
  grad_in_desc.set(grad_in_contiguous);

  auto pts_idx_of_voxels_impl =
      torch_mlu::getMluTensorImpl(pts_idx_of_voxels_contiguous);
  auto argmax_impl = torch_mlu::getMluTensorImpl(argmax_contiguous);
  auto grad_out_impl = torch_mlu::getMluTensorImpl(grad_out_contiguous);
  auto grad_in_impl = torch_mlu::getMluTensorImpl(grad_in_contiguous);

  auto pts_idx_of_voxels_ptr = pts_idx_of_voxels_impl->cnnlMalloc();
  auto argmax_ptr = argmax_impl->cnnlMalloc();
  auto grad_out_ptr = grad_out_impl->cnnlMalloc();
  auto grad_in_ptr = grad_in_impl->cnnlMalloc();

  CNLOG(INFO) << "Call mluOpRoiawarePool3dBackward().";
  TORCH_MLUOP_CHECK(mluOpRoiawarePool3dBackward(
      handle, pool_method, boxes_num, out_x, out_y, out_z, channels,
      max_pts_each_voxel, pts_idx_of_voxels_desc.desc(), pts_idx_of_voxels_ptr,
      argmax_desc.desc(), argmax_ptr, grad_out_desc.desc(), grad_out_ptr,
      grad_in_desc.desc(), grad_in_ptr));
}

void roiaware_pool3d_backward_mlu(int boxes_num, int out_x, int out_y,
                                  int out_z, int channels,
                                  int max_pts_each_voxel,
                                  const Tensor pts_idx_of_voxels,
                                  const Tensor argmax, const Tensor grad_out,
                                  Tensor grad_in, int pool_method) {
  RoiawarePool3dBackwardMLUKernelLauncher(
      pool_method, boxes_num, out_x, out_y, out_z, channels, max_pts_each_voxel,
      pts_idx_of_voxels, argmax, grad_out, grad_in);
}

void roiaware_pool3d_backward_impl(int boxes_num, int out_x, int out_y,
                                   int out_z, int channels,
                                   int max_pts_each_voxel,
                                   const Tensor pts_idx_of_voxels,
                                   const Tensor argmax, const Tensor grad_out,
                                   Tensor grad_in, int pool_method);

REGISTER_DEVICE_IMPL(roiaware_pool3d_backward_impl, MLU,
                     roiaware_pool3d_backward_mlu);
