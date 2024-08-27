#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

int hard_voxelize_forward_impl(const at::Tensor &points, at::Tensor &voxels,
                               at::Tensor &coors,
                               at::Tensor &num_points_per_voxel,
                               const std::vector<float> voxel_size,
                               const std::vector<float> coors_range,
                               const int max_points, const int max_voxels,
                               const int NDim = 3);

void dynamic_voxelize_forward_impl(const at::Tensor &points, at::Tensor &coors,
                                   const std::vector<float> voxel_size,
                                   const std::vector<float> coors_range,
                                   const int NDim = 3);

int hard_voxelize_forward_npu(const at::Tensor &points, at::Tensor &voxels,
                              at::Tensor &coors,
                              at::Tensor &num_points_per_voxel,
                              const std::vector<float> voxel_size,
                              const std::vector<float> coors_range,
                              const int max_points, const int max_voxels,
                              const int NDim = 3) {
  at::Tensor voxel_num_tmp = at::empty({1}, points.options());
  at::Tensor voxel_num = voxel_num_tmp.to(at::kInt);

  at::Tensor voxel_size_cpu = at::from_blob(
      const_cast<float *>(voxel_size.data()), {3}, dtype(at::kFloat));
  at::Tensor voxel_size_npu = voxel_size_cpu.to(points.device());

  at::Tensor coors_range_cpu = at::from_blob(
      const_cast<float *>(coors_range.data()), {6}, dtype(at::kFloat));
  at::Tensor coors_range_npu = coors_range_cpu.to(points.device());

  int64_t max_points_ = (int64_t)max_points;
  int64_t max_voxels_ = (int64_t)max_voxels;

  // only support true now
  bool deterministic = true;

  OpCommand cmd;
  cmd.Name("Voxelization")
      .Input(points)
      .Input(voxel_size_npu)
      .Input(coors_range_npu)
      .Output(voxels)
      .Output(coors)
      .Output(num_points_per_voxel)
      .Output(voxel_num)
      .Attr("max_points", max_points_)
      .Attr("max_voxels", max_voxels_)
      .Attr("deterministic", deterministic)
      .Run();
  auto voxel_num_cpu = voxel_num.to(at::kCPU);
  int voxel_num_int = voxel_num_cpu.data_ptr<int>()[0];
  return voxel_num_int;
}

void dynamic_voxelize_forward_npu(const at::Tensor &points, at::Tensor &coors,
                                  const std::vector<float> voxel_size,
                                  const std::vector<float> coors_range,
                                  const int NDim = 3) {
  uint32_t ptsNum = points.size(0);
  uint32_t ptsFeature = points.size(1);
  at::Tensor ptsTrans = at::transpose(points, 0, 1);
  double coors_min_x = coors_range[0];
  double coors_min_y = coors_range[1];
  double coors_min_z = coors_range[2];
  double coors_max_x = coors_range[3];
  double coors_max_y = coors_range[4];
  double coors_max_z = coors_range[5];
  double voxel_x = voxel_size[0];
  double voxel_y = voxel_size[1];
  double voxel_z = voxel_size[2];
  int grid_x = std::round((coors_max_x - coors_min_x) / voxel_x);
  int grid_y = std::round((coors_max_y - coors_min_y) / voxel_y);
  int grid_z = std::round((coors_max_z - coors_min_z) / voxel_z);

  at::Tensor tmp_coors =
      at::zeros({3, ptsNum}, points.options().dtype(at::kInt));
  EXEC_NPU_CMD(aclnnDynamicVoxelization, ptsTrans, coors_min_x, coors_min_y,
               coors_min_z, voxel_x, voxel_y, voxel_z, grid_x, grid_y, grid_z,
               tmp_coors);
  tmp_coors.transpose_(0, 1);
  coors.copy_(tmp_coors);
}

REGISTER_NPU_IMPL(hard_voxelize_forward_impl, hard_voxelize_forward_npu);
REGISTER_NPU_IMPL(dynamic_voxelize_forward_impl, dynamic_voxelize_forward_npu);
