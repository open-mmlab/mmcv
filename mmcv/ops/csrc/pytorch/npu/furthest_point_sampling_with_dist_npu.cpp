#include "pytorch_npu_helper.hpp"
using namespace NPU_NAME_SPACE;
using namespace std;

void furthest_point_sampling_with_dist_npu(Tensor points_tensor,
                                           Tensor temp_tensor,
                                           Tensor idx_tensor, int b, int n,
                                           int m) {
  TORCH_CHECK(
      (points_tensor.sizes()[1] >= m),
      "the num of sampled points should smaller than total num of points.");
  EXEC_NPU_CMD(aclnnFurthestPointSamplingWithDist, points_tensor, temp_tensor,
               m, idx_tensor);
}

void furthest_point_sampling_with_dist_forward_impl(Tensor points_tensor,
                                                    Tensor temp_tensor,
                                                    Tensor idx_tensor, int b,
                                                    int n, int m);

REGISTER_NPU_IMPL(furthest_point_sampling_with_dist_forward_impl,
                  furthest_point_sampling_with_dist_npu);
