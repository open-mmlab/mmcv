#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void furthest_point_sampling_forward_npu(Tensor points_tensor,
                                         Tensor temp_tensor, Tensor idx_tensor,
                                         int b, int n, int m) {
  TORCH_CHECK(
      (points_tensor.sizes()[1] >= m),
      "the num of sampled points should smaller than total num of points.");
  at::Tensor points_xyz = points_tensor.transpose(1, 2).contiguous();
  at::Tensor nearest_dist = temp_tensor.contiguous();
  EXEC_NPU_CMD(aclnnFurthestPointSampling, points_xyz, nearest_dist, m,
               idx_tensor);
}

void furthest_point_sampling_forward_impl(Tensor points_tensor,
                                          Tensor temp_tensor, Tensor idx_tensor,
                                          int b, int n, int m);

REGISTER_NPU_IMPL(furthest_point_sampling_forward_impl,
                  furthest_point_sampling_forward_npu);
