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
  at::Tensor features = trans_features.contiguous();
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
  at::Tensor res = output.contiguous();
  out.copy_(res);
}

void group_points_backward_npu(int b, int c, int n, int npoints, int nsample,
                               const Tensor grad_out, const Tensor idx,
                               Tensor grad_features) {
  at::Tensor trans_idx = idx.view({b * npoints * nsample});
  at::Tensor trans_grad_out = grad_out.permute({0, 2, 3, 1});
  at::Tensor grad_out_tensor = trans_grad_out.contiguous();
  grad_out_tensor = grad_out_tensor.view({b * npoints * nsample, c});
  at::Tensor out = at::zeros({b, n, c}, grad_out.options());

  EXEC_NPU_CMD(aclnnGroupPointsGrad, grad_out_tensor, trans_idx, b, c, n,
               npoints, nsample, out);

  at::Tensor grad_points = out.transpose(1, 2);

  grad_features.copy_(grad_points);
}

void group_points_forward_impl(int b, int c, int n, int npoints, int nsample,
                               const Tensor points, const Tensor idx,
                               Tensor out);
void group_points_backward_impl(int b, int c, int n, int npoints, int nsample,
                                const Tensor points, const Tensor idx,
                                Tensor out);

REGISTER_NPU_IMPL(group_points_forward_impl, group_points_forward_npu);
REGISTER_NPU_IMPL(group_points_backward_impl, group_points_backward_npu);
