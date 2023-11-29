#include "pytorch_npu_helper.hpp"
using namespace NPU_NAME_SPACE;
using namespace std;

void furthest_point_sampling_with_dist_npu(Tensor points_tensor,
                                           Tensor temp_tensor,
                                           Tensor idx_tensor, int b, int n,
                                           int m) {
  auto output_size = {b, m};
  at::Tensor result =
      at::empty(output_size, points_tensor.options().dtype(at::kInt));
  EXEC_NPU_CMD(aclnnFurthestPointSamplingWithDist, points_tensor, temp_tensor,
               m, result);
}

void furthest_point_sampling_with_dist_forward_impl(Tensor points_tensor,
                                                    Tensor temp_tensor,
                                                    Tensor idx_tensor, int b,
                                                    int n, int m);

REGISTER_NPU_IMPL(furthest_point_sampling_with_dist_forward_impl,
                  furthest_point_sampling_with_dist_npu);
