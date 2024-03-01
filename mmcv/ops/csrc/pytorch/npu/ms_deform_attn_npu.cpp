#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

Tensor ms_deform_attn_impl_forward(const Tensor &value,
                                   const Tensor &value_spatial_shapes,
                                   const Tensor &value_level_start_index,
                                   const Tensor &sampling_locations,
                                   const Tensor &attention_weights,
                                   const int im2col_step);

void check_support(const Tensor &value, const Tensor &attention_weights) {
  TORCH_CHECK(
      (value.scalar_type() == at::kFloat || value.scalar_type() == at::kHalf),
      "Dtype of value should be float32 or float16.");
  int64_t num_heads = value.size(2);
  int64_t embed_dims = value.size(3);
  int64_t num_points = attention_weights.size(4);
  TORCH_CHECK((num_heads >= 4 && num_heads <= 8),
              "num_heads should be in the range of [4, 8]");
  TORCH_CHECK((embed_dims >= 32 && embed_dims <= 256),
              "embed_dims should be in the range of [32, 256]");
  TORCH_CHECK((num_points >= 4 && num_points <= 8),
              "num_points should be in the range of [4, 8]");
}

Tensor ms_deform_attn_forward_npu(const Tensor &value,
                                  const Tensor &value_spatial_shapes,
                                  const Tensor &value_level_start_index,
                                  const Tensor &sampling_locations,
                                  const Tensor &attention_weights,
                                  const int im2col_step) {
  check_support(value, attention_weights);
  at::Tensor value_fp32 = value;
  at::Tensor value_spatial_shapes_int32 = value_spatial_shapes;
  at::Tensor value_level_start_index_int32 = value_level_start_index;
  at::Tensor sampling_locations_fp32 = sampling_locations;
  at::Tensor attention_weights_fp32 = attention_weights;
  if (value.scalar_type() != at::kFloat) {
    value_fp32 = value.to(at::kFloat);
  }
  if (value_spatial_shapes.scalar_type() != at::kInt) {
    value_spatial_shapes_int32 = value_spatial_shapes.to(at::kInt);
  }
  if (value_level_start_index.scalar_type() != at::kInt) {
    value_level_start_index_int32 = value_level_start_index.to(at::kInt);
  }
  if (sampling_locations.scalar_type() != at::kFloat) {
    sampling_locations_fp32 = sampling_locations.to(at::kFloat);
  }
  if (attention_weights.scalar_type() != at::kFloat) {
    attention_weights_fp32 = attention_weights.to(at::kFloat);
  }

  c10::SmallVector<int64_t, 3> output_size = {
      value.size(0), sampling_locations.size(1), value.size(2) * value.size(3)};
  at::Tensor output = at::empty(output_size, value_fp32.options());

  OpCommand cmd;
  cmd.Name("MultiScaleDeformableAttnFunction")
      .Input(value_fp32)
      .Input(value_spatial_shapes_int32)
      .Input(value_level_start_index_int32)
      .Input(sampling_locations_fp32)
      .Input(attention_weights_fp32)
      .Output(output)
      .Run();

  at::Tensor real_output = output;
  if (value.scalar_type() != at::kFloat) {
    real_output = output.to(value.scalar_type());
  }
  return real_output;
}

REGISTER_NPU_IMPL(ms_deform_attn_impl_forward, ms_deform_attn_forward_npu);
