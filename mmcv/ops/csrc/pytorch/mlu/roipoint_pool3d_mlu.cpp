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

void RoIPointPool3dForwardMLUKernelLauncher(
    int batch_size, int pts_num, int boxes_num, int feature_in_len,
    int sampled_pts_num, const Tensor xyz, const Tensor boxes3d,
    const Tensor pts_feature, Tensor pooled_features,
    Tensor pooled_empty_flag) {
  // check datatype
  TORCH_CHECK(((xyz.scalar_type() == pooled_features.scalar_type()) &&
               (boxes3d.scalar_type() == pooled_features.scalar_type()) &&
               (pts_feature.scalar_type() == pooled_features.scalar_type())),
              "data types of xyz, boxes3d, pts_feature and pooled_features "
              "should be the same, ",
              "but now xyz type is ", xyz.scalar_type(), ", boxes3d type is ",
              boxes3d.scalar_type(), ", pts_feature type is ",
              pts_feature.scalar_type(), ", pooled_features type is ",
              pooled_features.scalar_type(), ".");
  TORCH_CHECK(
      (xyz.scalar_type() == at::kFloat || xyz.scalar_type() == at::kHalf),
      "xyz type should be Float or Half, got ", xyz.scalar_type(), ".");
  TORCH_CHECK((pooled_empty_flag.scalar_type() == at::kInt),
              "pooled_empty_flag type should be Int, got ",
              pooled_empty_flag.scalar_type(), ".");

  // check shape
  TORCH_CHECK(boxes3d.dim() == 3, "boxes3d should be a 3d tensor, got ",
              boxes3d.dim(), "D.");
  TORCH_CHECK(pts_feature.dim() == 3, "pts_feature should be a 3d tensor, got ",
              pts_feature.dim(), "D.");

  TORCH_CHECK(boxes3d.size(2) == 7,
              "the 3rd dimensions of boxes3d should be 7, got ",
              boxes3d.size(2), ".");
  TORCH_CHECK((boxes3d.size(0) == batch_size),
              "the 1st dimensions of boxes3d should be batch_size, ",
              "but now the 1st dimension of boxes3d is ", boxes3d.size(0),
              ", and batch_size is ", batch_size, ".");
  TORCH_CHECK((pts_feature.size(0) == batch_size),
              "the 1st dimensions of pts_feature should be batch_size, ",
              "but now the 1st dimension of pts_feature is ",
              pts_feature.size(0), ", and batch_size is ", batch_size, ".");
  TORCH_CHECK((pts_feature.size(1) == pts_num),
              "the 2nd dimensions of pts_feature should be pts_num, ",
              "but now the 2nd dimension of pts_feature is ",
              pts_feature.size(1), ", and pts_num is ", pts_num, ".");

  // check zero element
  if (xyz.numel() == 0 || pts_feature.numel() == 0 || boxes3d.numel() == 0 ||
      pooled_features.numel() == 0 || pooled_empty_flag.numel() == 0) {
    return;
  }

  // large tensor check
  const size_t max_input_size = 2147483648;
  TORCH_CHECK(xyz.numel() < max_input_size,
              "xyz element num should be less than 2^31, got ", xyz.numel(),
              ".");
  TORCH_CHECK(boxes3d.numel() < max_input_size,
              "boxes3d element num should be less than 2^31, got ",
              boxes3d.numel(), ".");
  TORCH_CHECK(pts_feature.numel() < max_input_size,
              "pts_feature element num should be less than 2^31, got ",
              pts_feature.numel(), ".");

  // set contiguous
  auto xyz_contiguous =
      torch_mlu::cnnl::ops::cnnl_contiguous(xyz, xyz.suggest_memory_format());
  auto pts_feature_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      pts_feature, pts_feature.suggest_memory_format());
  auto boxes3d_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      boxes3d, boxes3d.suggest_memory_format());
  auto pooled_features_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      pooled_features, pooled_features.suggest_memory_format());
  auto pooled_empty_flag_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      pooled_empty_flag, pooled_empty_flag.suggest_memory_format());

  // get ptr of tensors
  auto xyz_impl = torch_mlu::getMluTensorImpl(xyz_contiguous);
  auto xyz_ptr = xyz_impl->cnnlMalloc();
  auto pts_feature_impl = torch_mlu::getMluTensorImpl(pts_feature_contiguous);
  auto pts_feature_ptr = pts_feature_impl->cnnlMalloc();
  auto boxes3d_impl = torch_mlu::getMluTensorImpl(boxes3d_contiguous);
  auto boxes3d_ptr = boxes3d_impl->cnnlMalloc();
  auto pooled_features_impl =
      torch_mlu::getMluTensorImpl(pooled_features_contiguous);
  auto pooled_features_ptr = pooled_features_impl->cnnlMalloc();
  auto pooled_empty_flag_impl =
      torch_mlu::getMluTensorImpl(pooled_empty_flag_contiguous);
  auto pooled_empty_flag_ptr = pooled_empty_flag_impl->cnnlMalloc();

  // create tensor descriptors
  MluOpTensorDescriptor xyz_desc, pts_feature_desc, boxes3d_desc,
      pooled_features_desc, pooled_empty_flag_desc;
  xyz_desc.set(xyz_contiguous);
  pts_feature_desc.set(pts_feature_contiguous);
  boxes3d_desc.set(boxes3d_contiguous);
  pooled_features_desc.set(pooled_features_contiguous);
  pooled_empty_flag_desc.set(pooled_empty_flag_contiguous);

  // get workspace
  size_t workspace_size = 0;
  auto handle = mluOpGetCurrentHandle();
  TORCH_MLUOP_CHECK(mluOpGetRoiPointPool3dWorkspaceSize(
      handle, batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num,
      xyz_desc.desc(), pts_feature_desc.desc(), boxes3d_desc.desc(),
      pooled_features_desc.desc(), pooled_empty_flag_desc.desc(),
      &workspace_size));

  auto workspace = at::empty(workspace_size, xyz.options().dtype(at::kByte));
  auto workspace_impl = torch_mlu::getMluTensorImpl(workspace);
  auto workspace_ptr = workspace_impl->cnnlMalloc();
  TORCH_MLUOP_CHECK(mluOpRoiPointPool3d(
      handle, batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num,
      xyz_desc.desc(), xyz_ptr, pts_feature_desc.desc(), pts_feature_ptr,
      boxes3d_desc.desc(), boxes3d_ptr, workspace_ptr, workspace_size,
      pooled_features_desc.desc(), pooled_features_ptr,
      pooled_empty_flag_desc.desc(), (int *)pooled_empty_flag_ptr));
}

void roipoint_pool3d_forward_mlu(int batch_size, int pts_num, int boxes_num,
                                 int feature_in_len, int sampled_pts_num,
                                 const Tensor xyz, const Tensor boxes3d,
                                 const Tensor pts_feature,
                                 Tensor pooled_features,
                                 Tensor pooled_empty_flag) {
  RoIPointPool3dForwardMLUKernelLauncher(
      batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num, xyz,
      boxes3d, pts_feature, pooled_features, pooled_empty_flag);
}

void roipoint_pool3d_forward_impl(int batch_size, int pts_num, int boxes_num,
                                  int feature_in_len, int sampled_pts_num,
                                  const Tensor xyz, const Tensor boxes3d,
                                  const Tensor pts_feature,
                                  Tensor pooled_features,
                                  Tensor pooled_empty_flag);

REGISTER_DEVICE_IMPL(roipoint_pool3d_forward_impl, MLU,
                     roipoint_pool3d_forward_mlu);
