#include "pytorch_npu_helper.hpp"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

using namespace NPU_NAME_SPACE;
using namespace std;

void three_interpolate_forward_npu(int b, int c, int m, int n,
                                   const Tensor points, const Tensor idx,
                                   const Tensor weight, Tensor out) {
  auto originDtype = points.scalar_type();
  TORCH_CHECK((originDtype == at::kFloat || originDtype == at::kHalf),
              "three_interpolate_forward ascend only support fp32 and fp16.");

  auto point_c_trans = points.transpose(1, 2);

  OpCommand cmd;
  cmd.Name("ThreeInterpolate")
      .Input(point_c_trans)
      .Input(idx)
      .Input(weight)
      .Output(out)
      .Run();

  auto output = out.view({b, n, c}).transpose(1, 2);
  auto res = output.contiguous();
  out.copy_(res);
}

void three_interpolate_backward_npu(int b, int c, int n, int m,
                                    const Tensor grad_out, const Tensor idx,
                                    const Tensor weight, Tensor grad_points) {
  auto originDtype = grad_out.scalar_type();
  TORCH_CHECK((originDtype == at::kFloat || originDtype == at::kHalf),
              "three_interpolate_backward ascend only support fp32 and fp16.");

  auto grad_x = at::unsqueeze(grad_out, 3);
  auto grad_y = at::unsqueeze(grad_points, 3);

  EXEC_NPU_CMD(aclnnThreeInterpolateBackward, grad_x, idx, weight, m, grad_y);

  auto output = at::squeeze(grad_y, 3);
  auto res = output.contiguous();
  grad_points.copy_(res);
}

void three_interpolate_forward_impl(int b, int c, int m, int n,
                                    const Tensor points, const Tensor idx,
                                    const Tensor weight, Tensor out);

void three_interpolate_backward_impl(int b, int c, int n, int m,
                                     const Tensor grad_out, const Tensor idx,
                                     const Tensor weight, Tensor grad_points);

REGISTER_NPU_IMPL(three_interpolate_forward_impl,
                  three_interpolate_forward_npu);

REGISTER_NPU_IMPL(three_interpolate_backward_impl,
                  three_interpolate_backward_npu);
