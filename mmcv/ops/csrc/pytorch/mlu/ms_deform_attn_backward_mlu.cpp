/*************************************************************************
 * Copyright (C) 2022 by Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "mlu_common_helper.h"
#include "pytorch_device_registry.hpp"
#include "pytorch_mlu_helper.hpp"

/*************************************************************************
 * This MACRO contains operations of simple tensor to mlu-tensor.
 * _contiguous, _desc, _impl, _ptr will be automatically generated in
 * this MACRO.
 *************************************************************************/
#define INITIAL_MLU_PARAM_WITH_TENSOR(NAME)                         \
  auto NAME##_contigous = torch_mlu::cnnl::ops::cnnl_contiguous(    \
      NAME, NAME.suggest_memory_format());                          \
  MluOpTensorDescriptor NAME##_desc;                                \
  NAME##_desc.set(NAME##_contigous);                                \
  auto NAME##_impl = torch_mlu::getMluTensorImpl(NAME##_contigous); \
  auto NAME##_ptr = NAME##_impl->cnnlMalloc();

void MsDeformAttnBackwardLauncher(
    const Tensor& value, const Tensor& spatial_shapes,
    const Tensor& level_start_index, const Tensor& sampling_loc,
    const Tensor& attn_weight, const Tensor& grad_output, Tensor& grad_value,
    Tensor& grad_sampling_loc, Tensor& grad_attn_weight,
    const int im2col_step) {
  auto handle = mluOpGetCurrentHandle();
  auto spatial_shapes_int = spatial_shapes.to(at::kInt);
  auto level_start_index_int = level_start_index.to(at::kInt);
  const int batch_size = value.size(0);
  const int num_heads = value.size(2);
  const int channels = value.size(3);
  const int num_queries = sampling_loc.size(1);

  auto grad_output_dim4 =
      grad_output.view({batch_size, num_queries, num_heads, channels});
  // auto grad_output_dim4 = grad_output.view({batch_size, num_queries,
  // num_heads, channels}).detach();
  INITIAL_MLU_PARAM_WITH_TENSOR(value);
  INITIAL_MLU_PARAM_WITH_TENSOR(spatial_shapes_int);
  INITIAL_MLU_PARAM_WITH_TENSOR(level_start_index_int);
  INITIAL_MLU_PARAM_WITH_TENSOR(sampling_loc);
  INITIAL_MLU_PARAM_WITH_TENSOR(attn_weight);
  INITIAL_MLU_PARAM_WITH_TENSOR(grad_output_dim4);
  // INITIAL_MLU_PARAM_WITH_TENSOR(grad_output);
  INITIAL_MLU_PARAM_WITH_TENSOR(grad_value);
  INITIAL_MLU_PARAM_WITH_TENSOR(grad_sampling_loc);
  INITIAL_MLU_PARAM_WITH_TENSOR(grad_attn_weight);

  mluOpMsDeformAttnBackward(
      handle, value_desc.desc(), value_ptr, spatial_shapes_int_desc.desc(),
      spatial_shapes_int_ptr, level_start_index_int_desc.desc(),
      level_start_index_int_ptr, sampling_loc_desc.desc(), sampling_loc_ptr,
      attn_weight_desc.desc(), attn_weight_ptr, grad_output_dim4_desc.desc(),
      grad_output_dim4_ptr, im2col_step, grad_value_desc.desc(), grad_value_ptr,
      grad_sampling_loc_desc.desc(), grad_sampling_loc_ptr,
      grad_attn_weight_desc.desc(), grad_attn_weight_ptr);

  return;
}

void ms_deform_attn_mlu_backward(
    const Tensor& value, const Tensor& spatial_shapes,
    const Tensor& level_start_index, const Tensor& sampling_loc,
    const Tensor& attn_weight, const Tensor& grad_output, Tensor& grad_value,
    Tensor& grad_sampling_loc, Tensor& grad_attn_weight,
    const int im2col_step) {
  return MsDeformAttnBackwardLauncher(value, spatial_shapes, level_start_index,
                                      sampling_loc, attn_weight, grad_output,
                                      grad_value, grad_sampling_loc,
                                      grad_attn_weight, im2col_step);
}

void ms_deform_attn_impl_backward(
    const Tensor& value, const Tensor& spatial_shapes,
    const Tensor& level_start_index, const Tensor& sampling_loc,
    const Tensor& attn_weight, const Tensor& grad_output, Tensor& grad_value,
    Tensor& grad_sampling_loc, Tensor& grad_attn_weight, const int im2col_step);

REGISTER_DEVICE_IMPL(ms_deform_attn_impl_backward, MLU,
                     ms_deform_attn_mlu_backward);
