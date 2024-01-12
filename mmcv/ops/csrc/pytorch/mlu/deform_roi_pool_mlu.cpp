/*************************************************************************
 * Copyright (C) 2022 Cambricon.
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

void DeformRoIPoolForwardMLUKernelLauncher(Tensor input, Tensor rois,
                                           Tensor offset, Tensor output,
                                           int pooled_height, int pooled_width,
                                           float spatial_scale,
                                           int sampling_ratio, float gamma) {
  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(input.dim());
  auto input_ = torch_mlu::cnnl::ops::cnnl_contiguous(input, memory_format);
  auto rois_contiguous =
      torch_mlu::cnnl::ops::cnnl_contiguous(rois, rois.suggest_memory_format());
  auto output_contiguous =
      torch_mlu::cnnl::ops::cnnl_contiguous(output, memory_format);

  MluOpTensorDescriptor input_desc, rois_desc, offset_desc, output_desc;
  input_desc.set_with_layout(input_, MLUOP_LAYOUT_NHWC);
  rois_desc.set(rois_contiguous);
  output_desc.set_with_layout(output_contiguous, MLUOP_LAYOUT_NHWC);

  mluOpTensorDescriptor_t offset_real_desc = NULL;
  void *offset_ptr = NULL;
  if (offset.defined() && offset.numel() > 0) {
    auto offset_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
        offset, offset.suggest_memory_format());
    offset_desc.set(offset_contiguous);
    offset_real_desc = offset_desc.desc();
    auto offset_impl = torch_mlu::getMluTensorImpl(offset_contiguous);
    offset_ptr = offset_impl->cnnlMalloc();
  }

  // get ptr of tensors
  auto input_impl = torch_mlu::getMluTensorImpl(input_);
  auto input_ptr = input_impl->cnnlMalloc();
  auto rois_impl = torch_mlu::getMluTensorImpl(rois_contiguous);
  auto rois_ptr = rois_impl->cnnlMalloc();
  auto output_impl = torch_mlu::getMluTensorImpl(output_contiguous);
  auto output_ptr = output_impl->cnnlMalloc();

  // get compute handle
  auto handle = mluOpGetCurrentHandle();
  TORCH_MLUOP_CHECK(mluOpDeformRoiPoolForward(
      handle, input_desc.desc(), input_ptr, rois_desc.desc(), rois_ptr,
      offset_real_desc, offset_ptr, pooled_height, pooled_width, spatial_scale,
      sampling_ratio, gamma, output_desc.desc(), output_ptr));

  output.copy_(output_contiguous);
}

void DeformRoIPoolBackwardMLUKernelLauncher(
    Tensor grad_output, Tensor input, Tensor rois, Tensor offset,
    Tensor grad_input, Tensor grad_offset, int pooled_height, int pooled_width,
    float spatial_scale, int sampling_ratio, float gamma) {
  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(grad_output.dim());
  auto grad_output_ =
      torch_mlu::cnnl::ops::cnnl_contiguous(grad_output, memory_format);
  memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(input.dim());
  auto input_ = torch_mlu::cnnl::ops::cnnl_contiguous(input, memory_format);
  auto rois_contiguous =
      torch_mlu::cnnl::ops::cnnl_contiguous(rois, rois.suggest_memory_format());
  auto grad_input_ =
      torch_mlu::cnnl::ops::cnnl_contiguous(grad_input, memory_format);

  // get ptr of tensors
  auto grad_output_impl = torch_mlu::getMluTensorImpl(grad_output_);
  auto grad_output_ptr = grad_output_impl->cnnlMalloc();
  auto input_impl = torch_mlu::getMluTensorImpl(input_);
  auto input_ptr = input_impl->cnnlMalloc();
  auto rois_impl = torch_mlu::getMluTensorImpl(rois_contiguous);
  auto rois_ptr = rois_impl->cnnlMalloc();
  auto grad_input_impl = torch_mlu::getMluTensorImpl(grad_input_);
  auto grad_input_ptr = grad_input_impl->cnnlMalloc();

  MluOpTensorDescriptor grad_output_desc, input_desc, rois_desc, offset_desc,
      grad_input_desc, grad_offset_desc;
  grad_output_desc.set_with_layout(grad_output_, MLUOP_LAYOUT_NHWC);
  input_desc.set_with_layout(input_, MLUOP_LAYOUT_NHWC);
  rois_desc.set(rois_contiguous);
  grad_input_desc.set_with_layout(grad_input_, MLUOP_LAYOUT_NHWC);
  mluOpTensorDescriptor_t offset_real_desc = NULL;
  void *offset_ptr = NULL;
  if (offset.defined() && offset.numel() > 0) {
    auto offset_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
        offset, offset.suggest_memory_format());
    offset_desc.set(offset_contiguous);
    offset_real_desc = offset_desc.desc();
    auto offset_impl = torch_mlu::getMluTensorImpl(offset_contiguous);
    offset_ptr = offset_impl->cnnlMalloc();
  }
  mluOpTensorDescriptor_t grad_offset_real_desc = NULL;
  void *grad_offset_ptr = NULL;
  if (grad_offset.defined() && grad_offset.numel() > 0) {
    auto grad_offset_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
        grad_offset, grad_offset.suggest_memory_format());
    grad_offset_desc.set(grad_offset_contiguous);
    grad_offset_real_desc = grad_offset_desc.desc();
    auto grad_offset_impl = torch_mlu::getMluTensorImpl(grad_offset_contiguous);
    grad_offset_ptr = grad_offset_impl->cnnlMalloc();
  }

  // get compute handle
  auto handle = mluOpGetCurrentHandle();
  TORCH_MLUOP_CHECK(mluOpDeformRoiPoolBackward(
      handle, grad_output_desc.desc(), grad_output_ptr, input_desc.desc(),
      input_ptr, rois_desc.desc(), rois_ptr, offset_real_desc, offset_ptr,
      pooled_height, pooled_width, spatial_scale, sampling_ratio, gamma,
      grad_input_desc.desc(), grad_input_ptr, grad_offset_real_desc,
      grad_offset_ptr));
  grad_input.copy_(grad_input_);
}

void deform_roi_pool_forward_mlu(Tensor input, Tensor rois, Tensor offset,
                                 Tensor output, int pooled_height,
                                 int pooled_width, float spatial_scale,
                                 int sampling_ratio, float gamma) {
  DeformRoIPoolForwardMLUKernelLauncher(input, rois, offset, output,
                                        pooled_height, pooled_width,
                                        spatial_scale, sampling_ratio, gamma);
}

void deform_roi_pool_backward_mlu(Tensor grad_output, Tensor input, Tensor rois,
                                  Tensor offset, Tensor grad_input,
                                  Tensor grad_offset, int pooled_height,
                                  int pooled_width, float spatial_scale,
                                  int sampling_ratio, float gamma) {
  DeformRoIPoolBackwardMLUKernelLauncher(
      grad_output, input, rois, offset, grad_input, grad_offset, pooled_height,
      pooled_width, spatial_scale, sampling_ratio, gamma);
}

void deform_roi_pool_forward_impl(Tensor input, Tensor rois, Tensor offset,
                                  Tensor output, int pooled_height,
                                  int pooled_width, float spatial_scale,
                                  int sampling_ratio, float gamma);

void deform_roi_pool_backward_impl(Tensor grad_output, Tensor input,
                                   Tensor rois, Tensor offset,
                                   Tensor grad_input, Tensor grad_offset,
                                   int pooled_height, int pooled_width,
                                   float spatial_scale, int sampling_ratio,
                                   float gamma);

REGISTER_DEVICE_IMPL(deform_roi_pool_forward_impl, MLU,
                     deform_roi_pool_forward_mlu);
REGISTER_DEVICE_IMPL(deform_roi_pool_backward_impl, MLU,
                     deform_roi_pool_backward_mlu);
