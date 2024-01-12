/*************************************************************************
 * Copyright (C) 2023 Cambricon.
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

std::vector<Tensor> dynamic_point_to_voxel_forward_mlu(
    const Tensor &feats, const Tensor &coors, const reduce_t reduce_type) {
  // params check
  TORCH_CHECK(feats.scalar_type() == at::kFloat,
              "feats type should be Float, got ", feats.scalar_type());
  TORCH_CHECK(coors.scalar_type() == at::kInt,
              "coors type should be Int32, got ", coors.scalar_type());
  TORCH_CHECK(feats.size(0) == coors.size(0),
              "feats.dim(0) and coors.dim(0) should be same, got ",
              feats.size(0), " vs ", coors.size(0));

  const int num_input = feats.size(0);
  const int num_feats = feats.size(1);
  // zero-element check
  if (num_input == 0)
    return {feats.clone().detach(), coors.clone().detach(),
            coors.new_empty({0}, torch::kInt32),
            coors.new_empty({0}, torch::kInt32)};

  auto mlu_reduce_type = getMluOpReduceMode(reduce_type);
  auto reduced_feats = at::empty({num_input, num_feats}, feats.options());
  auto out_coors = at::empty({num_input, 3}, coors.options());
  auto coors_map = at::empty({num_input}, coors.options());
  auto reduce_count = at::empty({num_input}, coors.options());
  auto voxel_num = at::empty({1}, coors.options());

  INITIAL_MLU_PARAM_WITH_TENSOR(feats);
  INITIAL_MLU_PARAM_WITH_TENSOR(coors);
  INITIAL_MLU_PARAM_WITH_TENSOR(reduced_feats);
  INITIAL_MLU_PARAM_WITH_TENSOR(out_coors);
  INITIAL_MLU_PARAM_WITH_TENSOR(coors_map);
  INITIAL_MLU_PARAM_WITH_TENSOR(reduce_count);
  INITIAL_MLU_PARAM_WITH_TENSOR(voxel_num);

  // get compute handle
  auto handle = mluOpGetCurrentHandle();

  size_t workspace_size;
  TORCH_MLUOP_CHECK(mluOpGetDynamicPointToVoxelForwardWorkspaceSize(
      handle, feats_desc.desc(), coors_desc.desc(), &workspace_size));
  auto workspace_tensor =
      at::empty(workspace_size, feats.options().dtype(at::kByte));
  INITIAL_MLU_PARAM_WITH_TENSOR(workspace_tensor);

  // launch kernel
  TORCH_MLUOP_CHECK(mluOpDynamicPointToVoxelForward(
      handle, mlu_reduce_type, feats_desc.desc(), feats_ptr, coors_desc.desc(),
      coors_ptr, workspace_tensor_ptr, workspace_size,
      reduced_feats_desc.desc(), reduced_feats_ptr, out_coors_desc.desc(),
      out_coors_ptr, coors_map_desc.desc(), coors_map_ptr,
      reduce_count_desc.desc(), reduce_count_ptr, voxel_num_desc.desc(),
      voxel_num_ptr));

  int voxel_num_value = *static_cast<int *>(voxel_num.cpu().data_ptr());
  TORCH_CHECK(voxel_num_value <= feats.size(0),
              "voxel_num should be less than or equal to feats_num, got ",
              voxel_num_value, " vs ", feats.size(0));
  return {reduced_feats.slice(0, 0, voxel_num_value),
          out_coors.slice(0, 0, voxel_num_value), coors_map,
          reduce_count.slice(0, 0, voxel_num_value)};
}

void dynamic_point_to_voxel_backward_mlu(
    Tensor &grad_feats, const Tensor &grad_reduced_feats, const Tensor &feats,
    const Tensor &reduced_feats, const Tensor &coors_idx,
    const Tensor &reduce_count, const reduce_t reduce_type) {
  // params check
  TORCH_CHECK(grad_reduced_feats.scalar_type() == at::kFloat,
              "grad_reduced_feats type should be Float, got ",
              grad_reduced_feats.scalar_type());
  TORCH_CHECK(feats.scalar_type() == at::kFloat,
              "feats type should be Float, got ", feats.scalar_type());
  TORCH_CHECK(reduced_feats.scalar_type() == at::kFloat,
              "reduced_feats type should be Float, got ",
              reduced_feats.scalar_type());
  TORCH_CHECK(coors_idx.scalar_type() == at::kInt,
              "coors_idx type should be Int32, got ", coors_idx.scalar_type());
  TORCH_CHECK(reduce_count.scalar_type() == at::kInt,
              "reduce_count type should be Int32, got ",
              reduce_count.scalar_type());

  const int num_input = feats.size(0);
  const int num_reduced = reduced_feats.size(0);
  const int num_feats = feats.size(1);

  grad_feats.fill_(0);

  // zero-element check
  if (num_input == 0 || num_reduced == 0) return;

  // TODO(miaochen): remove this after mlu-ops supports other mode of reduce.
  TORCH_CHECK(reduce_type == reduce_t::MAX,
              "only supports max reduce in current version, got ",
              to_string(reduce_type));

  int voxel_num_value = reduced_feats.size(0);
  auto opts = torch::TensorOptions().dtype(torch::kInt32);
  auto voxel_num =
      torch::from_blob(&voxel_num_value, {1}, opts).clone().to(at::kMLU);
  auto mlu_reduce_type = getMluOpReduceMode(reduce_type);

  INITIAL_MLU_PARAM_WITH_TENSOR(grad_feats);
  INITIAL_MLU_PARAM_WITH_TENSOR(grad_reduced_feats);
  INITIAL_MLU_PARAM_WITH_TENSOR(feats);
  INITIAL_MLU_PARAM_WITH_TENSOR(reduced_feats);
  INITIAL_MLU_PARAM_WITH_TENSOR(coors_idx);
  INITIAL_MLU_PARAM_WITH_TENSOR(reduce_count);
  INITIAL_MLU_PARAM_WITH_TENSOR(voxel_num);

  // get compute handle
  auto handle = mluOpGetCurrentHandle();

  size_t workspace_size;
  TORCH_MLUOP_CHECK(mluOpGetDynamicPointToVoxelBackwardWorkspaceSize(
      handle, mlu_reduce_type, grad_feats_desc.desc(), feats_desc.desc(),
      grad_reduced_feats_desc.desc(), coors_idx_desc.desc(),
      reduce_count_desc.desc(), voxel_num_desc.desc(), &workspace_size));
  auto workspace_tensor =
      at::empty(workspace_size, feats.options().dtype(at::kByte));
  INITIAL_MLU_PARAM_WITH_TENSOR(workspace_tensor);

  // launch kernel
  TORCH_MLUOP_CHECK(mluOpDynamicPointToVoxelBackward(
      handle, mlu_reduce_type, grad_reduced_feats_desc.desc(),
      grad_reduced_feats_ptr, feats_desc.desc(), feats_ptr,
      reduced_feats_desc.desc(), reduced_feats_ptr, coors_idx_desc.desc(),
      coors_idx_ptr, reduce_count_desc.desc(), reduce_count_ptr,
      voxel_num_desc.desc(), voxel_num_ptr, workspace_tensor_ptr,
      workspace_size, grad_feats_desc.desc(), grad_feats_ptr));
}

std::vector<Tensor> dynamic_point_to_voxel_forward_impl(
    const Tensor &feats, const Tensor &coors, const reduce_t reduce_type);

void dynamic_point_to_voxel_backward_impl(
    Tensor &grad_feats, const Tensor &grad_reduced_feats, const Tensor &feats,
    const Tensor &reduced_feats, const Tensor &coors_idx,
    const Tensor &reduce_count, const reduce_t reduce_type);

REGISTER_DEVICE_IMPL(dynamic_point_to_voxel_forward_impl, MLU,
                     dynamic_point_to_voxel_forward_mlu);
REGISTER_DEVICE_IMPL(dynamic_point_to_voxel_backward_impl, MLU,
                     dynamic_point_to_voxel_backward_mlu);
