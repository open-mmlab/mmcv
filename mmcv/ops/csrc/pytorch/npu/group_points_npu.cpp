#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void group_points_forward_npu(int b, int c, int n, int npoints, int nsample,
                              const Tensor points, const Tensor idx,
                              Tensor out) {
  // b, c, n, and npoints do not need to be passed into gatherv2,
  // b, c, n, and npoints are calculated inside the operator
  // gatherv2 operator in ascend needs to set axis to 0, batch_dims is 0
  c10::SmallVector<int64_t, N> axis = {0};
  int64_t batch_dims = 0;

  auto index = at::arange(0, b);
  index = index.to(points.device());
  index = index.view({-1, 1, 1});
  index = at::mul(index, n);
  at::Tensor indices = at::add(index, idx);
  indices = indices.view({-1});

  at::Tensor trans_features = points.transpose(1, 2);
  at::Tensor features = NpuUtils::format_contiguous(trans_features);
  features = features.view({b * n, c});

  OpCommand cmd;
  cmd.Name("GatherV2")
      .Input(features)
      .Input(indices)
      .Input(axis)
      .Output(out)
      .Attr("batch_dims", batch_dims)
      .Run();

  at::Tensor output =
      out.view({b, npoints, nsample, c}).transpose(1, 3).transpose(2, 3);
  at::Tensor res = NpuUtils::format_contiguous(output);
  out.copy_(res);
}

void group_points_forward_impl(int b, int c, int n, int npoints, int nsample,
                               const Tensor points, const Tensor idx,
                               Tensor out);

REGISTER_NPU_IMPL(group_points_forward_impl, group_points_forward_npu);
