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

/*************************************************************************
 * This MACRO contains operations of simple tensor to mlu-tensor.
 * _contiguous, _desc, _impl, _ptr will be automatically generated in
 * this MACRO.
 *************************************************************************/
#define INITIAL_MLU_PARAM_WITH_TENSOR(NAME)                         \
  auto NAME##_contigous = torch_mlu::cnnl::ops::cnnl_contiguous(    \
      NAME, NAME.suggest_memory_format());                          \
  MluOpTensorDescriptor NAME##_desc;                                \
  NAME##_desc.set(NAME##_contigous);                                \
  auto NAME##_impl = torch_mlu::getMluTensorImpl(NAME##_contigous); \
  auto NAME##_ptr = NAME##_impl->cnnlMalloc();

int HardVoxelizeForwardMLUKernelLauncher(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim = 3) {
  std::vector<float> _voxel_size(voxel_size.begin(), voxel_size.end());
  std::vector<float> _coors_range(coors_range.begin(), coors_range.end());
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  auto voxel_size_tensor =
      torch::from_blob(_voxel_size.data(), {int64_t(_voxel_size.size())}, opts)
          .clone()
          .to(at::kMLU);
  auto coors_range_tensor =
      torch::from_blob(_coors_range.data(), {int64_t(_coors_range.size())},
                       opts)
          .clone()
          .to(at::kMLU);
  INITIAL_MLU_PARAM_WITH_TENSOR(points);
  INITIAL_MLU_PARAM_WITH_TENSOR(voxels);
  INITIAL_MLU_PARAM_WITH_TENSOR(coors);
  INITIAL_MLU_PARAM_WITH_TENSOR(num_points_per_voxel);
  INITIAL_MLU_PARAM_WITH_TENSOR(voxel_size_tensor);
  INITIAL_MLU_PARAM_WITH_TENSOR(coors_range_tensor);

  auto voxel_num_tensor = at::empty({1}, points.options().dtype(torch::kInt32));
  INITIAL_MLU_PARAM_WITH_TENSOR(voxel_num_tensor);

  size_t workspace_size;
  auto handle = mluOpGetCurrentHandle();
  TORCH_MLUOP_CHECK(mluOpGetVoxelizationWorkspaceSize(
      handle, points_desc.desc(), voxel_size_tensor_desc.desc(),
      coors_range_tensor_desc.desc(), max_points, max_voxels, NDim, true,
      voxels_desc.desc(), coors_desc.desc(), num_points_per_voxel_desc.desc(),
      voxel_num_tensor_desc.desc(), &workspace_size));
  auto workspace_tensor =
      at::empty(workspace_size, points.options().dtype(at::kByte));
  INITIAL_MLU_PARAM_WITH_TENSOR(workspace_tensor);

  TORCH_MLUOP_CHECK(mluOpVoxelization(
      handle, points_desc.desc(), points_ptr, voxel_size_tensor_desc.desc(),
      voxel_size_tensor_ptr, coors_range_tensor_desc.desc(),
      coors_range_tensor_ptr, max_points, max_voxels, NDim, true,
      workspace_tensor_ptr, workspace_size, voxels_desc.desc(), voxels_ptr,
      coors_desc.desc(), coors_ptr, num_points_per_voxel_desc.desc(),
      num_points_per_voxel_ptr, voxel_num_tensor_desc.desc(),
      voxel_num_tensor_ptr));
  auto voxel_num_cpu = voxel_num_tensor.to(at::kCPU);
  int voxel_num_int = voxel_num_cpu.data_ptr<int>()[0];
  return voxel_num_int;
}

int hard_voxelize_forward_mlu(const at::Tensor &points, at::Tensor &voxels,
                              at::Tensor &coors,
                              at::Tensor &num_points_per_voxel,
                              const std::vector<float> voxel_size,
                              const std::vector<float> coors_range,
                              const int max_points, const int max_voxels,
                              const int NDim) {
  return HardVoxelizeForwardMLUKernelLauncher(
      points, voxels, coors, num_points_per_voxel, voxel_size, coors_range,
      max_points, max_voxels, NDim);
}

int hard_voxelize_forward_impl(const at::Tensor &points, at::Tensor &voxels,
                               at::Tensor &coors,
                               at::Tensor &num_points_per_voxel,
                               const std::vector<float> voxel_size,
                               const std::vector<float> coors_range,
                               const int max_points, const int max_voxels,
                               const int NDim);

REGISTER_DEVICE_IMPL(hard_voxelize_forward_impl, MLU,
                     hard_voxelize_forward_mlu);
