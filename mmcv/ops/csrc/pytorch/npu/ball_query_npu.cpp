#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void ball_query_forward_npu(int b, int n, int m, float min_radius,
                            float max_radius, int nsample, const Tensor new_xyz,
                            const Tensor xyz, Tensor idx) {
  int64_t nsample_i64 = nsample;

  // transpose new_xyz from [B, M, 3] to [M, B, 3]
  at::Tensor new_xyz_transpose = new_xyz.transpose(0, 1).to(at::kFloat);

  // transpose xyz from [B, N, 3] to [B, 3, N]
  at::Tensor xyz_transpose = xyz.transpose(1, 2).to(at::kFloat);

  // transpose idx from [B, M, nsample] to [M, B, nsample]
  at::Tensor idx_transpose = idx.transpose(0, 1).contiguous();

  OpCommand cmd;
  cmd.Name("BallQuery")
      .Input(xyz_transpose)
      .Input(new_xyz_transpose)
      .Output(idx_transpose)
      .Attr("min_radius", min_radius)
      .Attr("max_radius", max_radius)
      .Attr("sample_num", nsample_i64)
      .Run();

  idx_transpose = idx_transpose.transpose(0, 1).contiguous();
  idx.copy_(idx_transpose);
}

void ball_query_forward_impl(int b, int n, int m, float min_radius,
                             float max_radius, int nsample,
                             const Tensor new_xyz, const Tensor xyz,
                             Tensor idx);

REGISTER_NPU_IMPL(ball_query_forward_impl, ball_query_forward_npu);
