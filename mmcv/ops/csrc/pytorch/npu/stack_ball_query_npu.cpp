#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void stack_ball_query_forward_npu(float max_radius, int nsample,
                                  const Tensor new_xyz,
                                  const Tensor new_xyz_batch_cnt,
                                  const Tensor xyz, const Tensor xyz_batch_cnt,
                                  Tensor idx) {
  at::Tensor xyz_transpose = xyz.transpose(0, 1).contiguous();
  double max_radius_double = double(max_radius);
  EXEC_NPU_CMD(aclnnStackBallQuery, xyz_transpose, new_xyz, xyz_batch_cnt,
               new_xyz_batch_cnt, max_radius_double, nsample, idx);
}

void stack_ball_query_forward_impl(float max_radius, int nsample,
                                   const Tensor new_xyz,
                                   const Tensor new_xyz_batch_cnt,
                                   const Tensor xyz, const Tensor xyz_batch_cnt,
                                   Tensor idx);

REGISTER_NPU_IMPL(stack_ball_query_forward_impl, stack_ball_query_forward_npu);
