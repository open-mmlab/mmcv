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

void ROIAlignRotatedForwardMLUKernelLauncher(Tensor input, Tensor rois,
                                             Tensor output, int pooled_height,
                                             int pooled_width,
                                             float spatial_scale,
                                             int sampling_ratio, bool aligned,
                                             bool clockwise) {
  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(input.dim());
  auto input_ = torch_mlu::cnnl::ops::cnnl_contiguous(input, memory_format);
  auto rois_contiguous =
      torch_mlu::cnnl::ops::cnnl_contiguous(rois, rois.suggest_memory_format());
  auto output_contiguous =
      torch_mlu::cnnl::ops::cnnl_contiguous(output, memory_format);

  MluOpTensorDescriptor input_desc, rois_desc, output_desc;
  input_desc.set_with_layout(input_, MLUOP_LAYOUT_NHWC);
  rois_desc.set(rois_contiguous);
  output_desc.set_with_layout(output_contiguous, MLUOP_LAYOUT_NHWC);

  // get ptr of tensors
  auto input_impl = torch_mlu::getMluTensorImpl(input_);
  auto input_ptr = input_impl->cnnlMalloc();
  auto rois_impl = torch_mlu::getMluTensorImpl(rois_contiguous);
  auto rois_ptr = rois_impl->cnnlMalloc();
  auto output_impl = torch_mlu::getMluTensorImpl(output_contiguous);
  auto output_ptr = output_impl->cnnlMalloc();

  // get compute handle
  auto handle = mluOpGetCurrentHandle();
  TORCH_MLUOP_CHECK(mluOpRoiAlignRotatedForward(
      handle, input_desc.desc(), input_ptr, rois_desc.desc(), rois_ptr,
      pooled_height, pooled_width, sampling_ratio, spatial_scale, aligned,
      clockwise, output_desc.desc(), output_ptr));

  output.copy_(output_contiguous);
}

void ROIAlignRotatedBackwardMLUKernelLauncher(
    Tensor top_grad, Tensor rois, Tensor bottom_grad, int pooled_height,
    int pooled_width, float spatial_scale, int sampling_ratio, bool aligned,
    bool clockwise) {
  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(top_grad.dim());
  auto top_grad_ =
      torch_mlu::cnnl::ops::cnnl_contiguous(top_grad, memory_format);
  auto rois_contiguous =
      torch_mlu::cnnl::ops::cnnl_contiguous(rois, rois.suggest_memory_format());
  auto bottom_grad_ =
      torch_mlu::cnnl::ops::cnnl_contiguous(bottom_grad, memory_format);

  // get ptr of tensors
  auto top_grad_impl = torch_mlu::getMluTensorImpl(top_grad_);
  auto top_grad_ptr = top_grad_impl->cnnlMalloc();
  auto rois_impl = torch_mlu::getMluTensorImpl(rois_contiguous);
  auto rois_ptr = rois_impl->cnnlMalloc();
  auto bottom_grad_impl = torch_mlu::getMluTensorImpl(bottom_grad_);
  auto bottom_grad_ptr = bottom_grad_impl->cnnlMalloc();

  MluOpTensorDescriptor top_grad_desc, rois_desc, bottom_grad_desc;
  top_grad_desc.set_with_layout(top_grad_, MLUOP_LAYOUT_NHWC);
  rois_desc.set(rois_contiguous);
  bottom_grad_desc.set_with_layout(bottom_grad_, MLUOP_LAYOUT_NHWC);

  // get compute handle
  auto handle = mluOpGetCurrentHandle();
  TORCH_MLUOP_CHECK(mluOpRoiAlignRotatedBackward(
      handle, top_grad_desc.desc(), top_grad_ptr, rois_desc.desc(), rois_ptr,
      pooled_height, pooled_width, sampling_ratio, spatial_scale, aligned,
      clockwise, bottom_grad_desc.desc(), bottom_grad_ptr));
  bottom_grad.copy_(bottom_grad_);
}

void roi_align_rotated_forward_mlu(Tensor input, Tensor rois, Tensor output,
                                   int aligned_height, int aligned_width,
                                   float spatial_scale, int sampling_ratio,
                                   bool aligned, bool clockwise) {
  ROIAlignRotatedForwardMLUKernelLauncher(input, rois, output, aligned_height,
                                          aligned_width, spatial_scale,
                                          sampling_ratio, aligned, clockwise);
}

void roi_align_rotated_backward_mlu(Tensor top_grad, Tensor rois,
                                    Tensor bottom_grad, int aligned_height,
                                    int aligned_width, float spatial_scale,
                                    int sampling_ratio, bool aligned,
                                    bool clockwise) {
  ROIAlignRotatedBackwardMLUKernelLauncher(
      top_grad, rois, bottom_grad, aligned_height, aligned_width, spatial_scale,
      sampling_ratio, aligned, clockwise);
}

void roi_align_rotated_forward_impl(Tensor input, Tensor rois, Tensor output,
                                    int aligned_height, int aligned_width,
                                    float spatial_scale, int sampling_ratio,
                                    bool aligned, bool clockwise);

void roi_align_rotated_backward_impl(Tensor top_grad, Tensor rois,
                                     Tensor bottom_grad, int aligned_height,
                                     int aligned_width, float spatial_scale,
                                     int sampling_ratio, bool aligned,
                                     bool clockwise);

REGISTER_DEVICE_IMPL(roi_align_rotated_forward_impl, MLU,
                     roi_align_rotated_forward_mlu);
REGISTER_DEVICE_IMPL(roi_align_rotated_backward_impl, MLU,
                     roi_align_rotated_backward_mlu);
