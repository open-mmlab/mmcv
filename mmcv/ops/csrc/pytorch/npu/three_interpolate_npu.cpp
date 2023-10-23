#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void three_interpolate_forward_npu(int b, int c, int m, int n,
                                   const Tensor points, const Tensor idx,
                                   const Tensor weight, Tensor out) {
  auto originDtype = points.scalar_type();
  at::Tensor pointsCast = points;
  at::Tensor weightCast = weight;
  at::Tensor outCast = out;

  TORCH_CHECK((originDtype == at::kFloat || originDtype == at::kHalf),
              "three_interpolate_forward ascend only support fp32 and fp16.");

  if (originDtype == at::ScalarType::Half) {
    pointsCast = points.to(at::kFloat);
    weightCast = weight.to(at::kFloat);
    outCast = out.to(at::kFloat);
  }

  auto point_c_trans = pointsCast.transpose(1, 2);

  OpCommand cmd;
  cmd.Name("ThreeInterpolate")
      .Input(point_c_trans)
      .Input(idx)
      .Input(weightCast)
      .Output(outCast)
      .Run();

  auto output = outCast.view({b, n, c}).transpose(1, 2);
  auto res = NpuUtils::format_contiguous(output);
  out.copy_(res);
}

void three_interpolate_backward_npu(int b, int c, int n, int m,
                                    const Tensor grad_out, const Tensor idx,
                                    const Tensor weight, Tensor grad_points) {
  auto originDtype = grad_out.scalar_type();
  at::Tensor gradOutCast = grad_out;
  at::Tensor weightCast = weight;
  at::Tensor gradPointsCast = grad_points;

  TORCH_CHECK((originDtype == at::kFloat || originDtype == at::kHalf),
              "three_interpolate_backward ascend only support fp32 and fp16.");

  if (originDtype == at::ScalarType::Half) {
    gradOutCast = grad_out.to(at::kFloat);
    weightCast = weight.to(at::kFloat);
    gradPointsCast = grad_points.to(at::kFloat);
  }

  OpCommand cmd;
  cmd.Name("ThreeInterpolateBackward")
      .Input(gradOutCast)
      .Input(idx)
      .Input(weightCast)
      .Output(gradPointsCast)
      .Attr("m", m)
      .Run();
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
