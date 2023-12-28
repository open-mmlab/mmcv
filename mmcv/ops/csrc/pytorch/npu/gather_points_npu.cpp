#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void gather_points_forward_npu(int b, int c, int n, int npoints,
                               const Tensor points, const Tensor idx,
                               Tensor out) {
  // b, c, n, and npoints do not need to be passed into gatherv2,
  // b, c, n, and npoints are calculated inside the operator
  // gatherv2 operator in ascend needs to set axis to 2, batch_dims is 1
  c10::SmallVector<int64_t, N> axis = {2};
  int64_t batch_dims = 1;

  OpCommand cmd;
  cmd.Name("GatherV2")
      .Input(points)
      .Input(idx)
      .Input(axis)
      .Output(out)
      .Attr("batch_dims", batch_dims)
      .Run();
}
void gather_points_backward_npu(int b, int c, int n, int npoints,
                                const Tensor grad_out, const Tensor idx,
                                Tensor grad_points) {
  at::Tensor indices = idx;
  if (idx.scalar_type() != at::ScalarType::Int) {
    indices = idx.to(at::kInt);
  }
  if (idx.dim() == 0) {
    indices.unsqueeze_(0);
  }
  int64_t dim = 0;
  auto shape = idx.sizes();
  c10::SmallVector<int64_t, 8> pad_size;
  for (uint64_t i = 0; i < shape.size(); i++) {
    pad_size.emplace_back(shape[i]);
  }
  at::Tensor trans_grad_points = grad_points.transpose(1, 2).contiguous();
  at::Tensor grad_points_view = trans_grad_points.view(
      {trans_grad_points.sizes()[0] * trans_grad_points.sizes()[1],
       trans_grad_points.sizes()[2]});
  at::Tensor trans_grad_out = grad_out.transpose(1, 2).contiguous();
  trans_grad_out = trans_grad_out.view(
      {trans_grad_out.sizes()[0] * trans_grad_out.sizes()[1],
       trans_grad_out.sizes()[2]});
  auto index = at::arange(0, b);
  index = index.to(grad_out.device());
  index = at::mul(index, n);
  index = index.view({b, 1});
  index = at::broadcast_to(index, pad_size);
  indices = at::add(index, indices);
  indices = indices.view({-1});
  OpCommand cmd;
  cmd.Name("InplaceIndexAdd")
      .Input(grad_points_view)
      .Input(indices)
      .Input(trans_grad_out)
      .Output(grad_points_view)
      .Attr("axis", dim)
      .Run();
  at::Tensor grad_points_result =
      grad_points_view.view(trans_grad_points.sizes());
  grad_points_result = grad_points_result.transpose(1, 2);
  grad_points.copy_(grad_points_result);
}

void gather_points_forward_impl(int b, int c, int n, int npoints,
                                const Tensor points, const Tensor idx,
                                Tensor out);
void gather_points_backward_impl(int b, int c, int n, int npoints,
                                 const Tensor grad_out, const Tensor idx,
                                 Tensor grad_points);

REGISTER_NPU_IMPL(gather_points_forward_impl, gather_points_forward_npu);
REGISTER_NPU_IMPL(gather_points_backward_impl, gather_points_backward_npu);
