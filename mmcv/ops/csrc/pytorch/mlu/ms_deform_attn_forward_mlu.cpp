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
#include "pytorch_device_registry.hpp"
#include "pytorch_mlu_helper.hpp"
#include "mlu_common_helper.h"

/*************************************************************************
 * This MACRO contains operations of simple tensor to mlu-tensor.
 * _contiguous, _desc, _impl, _ptr will be automatically generated in
 * this MACRO.
 *************************************************************************/
#define INITIAL_MLU_PARAM_WITH_TENSOR(NAME)                           \
    auto NAME##_contigous = torch_mlu::cnnl::ops::cnnl_contiguous(    \
        NAME, NAME.suggest_memory_format());                          \
    MluOpTensorDescriptor NAME##_desc;                                \
    NAME##_desc.set(NAME##_contigous);                                \
    auto NAME##_impl = torch_mlu::getMluTensorImpl(NAME##_contigous); \
    auto NAME##_ptr = NAME##_impl->cnnlMalloc();

Tensor MsDeformAttnForwardLauncher(const Tensor& value,
                                   const Tensor& spatial_shapes,
                                   const Tensor& level_start_index,
                                   const Tensor& sampling_loc,
                                   const Tensor& attn_weight,
                                   const int im2col_step) {
  auto handle = mluOpGetCurrentHandle();
  const int batch_size = value.size(0);
  const int num_heads = value.size(2);
  const int channels = value.size(3);
  const int num_queries = sampling_loc.size(1);
  auto output = at::zeros({batch_size, num_queries, num_heads, channels},
                          value.options());
  auto spatial_shapes_int = spatial_shapes.to(at::kInt);
  auto level_start_index_int = level_start_index.to(at::kInt);
  INITIAL_MLU_PARAM_WITH_TENSOR(output);
  INITIAL_MLU_PARAM_WITH_TENSOR(value);
  INITIAL_MLU_PARAM_WITH_TENSOR(spatial_shapes_int);
  INITIAL_MLU_PARAM_WITH_TENSOR(level_start_index_int);
  INITIAL_MLU_PARAM_WITH_TENSOR(sampling_loc);
  INITIAL_MLU_PARAM_WITH_TENSOR(attn_weight);

  mluOpMsDeformAttnForward(
      handle, value_desc.desc(), value_ptr,
      spatial_shapes_int_desc.desc(), spatial_shapes_int_ptr,
      level_start_index_int_desc.desc(), level_start_index_int_ptr,
      sampling_loc_desc.desc(), sampling_loc_ptr,
      attn_weight_desc.desc(), attn_weight_ptr,
      im2col_step, output_desc.desc(), output_ptr);

  output = output.view({batch_size, num_queries, num_heads * channels});
  return output;
}

Tensor ms_deform_attn_mlu_forward(const Tensor& value,
                                  const Tensor& spatial_shapes,
                                  const Tensor& level_start_index,
                                  const Tensor& sampling_loc,
                                  const Tensor& attn_weight,
                                  const int im2col_step) {
  return MsDeformAttnForwardLauncher(
      value, spatial_shapes, level_start_index, sampling_loc,
      attn_weight, im2col_step);
}

Tensor ms_deform_attn_impl_forward(const Tensor& value,
                                   const Tensor& spatial_shapes,
                                   const Tensor& level_start_index,
                                   const Tensor& sampling_loc,
                                   const Tensor& attn_weight,
                                   const int im2col_step);

REGISTER_DEVICE_IMPL(ms_deform_attn_impl_forward, MLU,
                     ms_deform_attn_mlu_forward);
