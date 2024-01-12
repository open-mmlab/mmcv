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

void RotatedFeatureAlignForwardMLUKernelLauncher(const Tensor features,
                                                 const Tensor best_bboxes,
                                                 const float spatial_scale,
                                                 const int points,
                                                 Tensor output) {
  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(features.dim());
  auto features_ =
      torch_mlu::cnnl::ops::cnnl_contiguous(features, memory_format);
  auto best_bboxes_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      best_bboxes, best_bboxes.suggest_memory_format());
  auto output_contiguous =
      torch_mlu::cnnl::ops::cnnl_contiguous(output, memory_format);

  MluOpTensorDescriptor features_desc, best_bboxes_desc, output_desc;
  features_desc.set_with_layout(features_, MLUOP_LAYOUT_NHWC);
  best_bboxes_desc.set(best_bboxes_contiguous);
  output_desc.set_with_layout(output_contiguous, MLUOP_LAYOUT_NHWC);

  // get ptr of tensors
  auto features_impl = torch_mlu::getMluTensorImpl(features_);
  auto features_ptr = features_impl->cnnlMalloc();
  auto best_bboxes_impl = torch_mlu::getMluTensorImpl(best_bboxes_contiguous);
  auto best_bboxes_ptr = best_bboxes_impl->cnnlMalloc();
  auto output_impl = torch_mlu::getMluTensorImpl(output_contiguous);
  auto output_ptr = output_impl->cnnlMalloc();

  // get compute handle
  auto handle = mluOpGetCurrentHandle();
  TORCH_MLUOP_CHECK(mluOpRotatedFeatureAlignForward(
      handle, features_desc.desc(), features_ptr, best_bboxes_desc.desc(),
      best_bboxes_ptr, spatial_scale, points, output_desc.desc(), output_ptr));

  output.copy_(output_contiguous);
}

void RotatedFeatureAlignBackwardMLUKernelLauncher(const Tensor top_grad,
                                                  const Tensor best_bboxes,
                                                  const float spatial_scale,
                                                  const int points,
                                                  Tensor bottom_grad) {
  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(top_grad.dim());
  auto top_grad_ =
      torch_mlu::cnnl::ops::cnnl_contiguous(top_grad, memory_format);
  auto best_bboxes_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      best_bboxes, best_bboxes.suggest_memory_format());
  auto bottom_grad_ =
      torch_mlu::cnnl::ops::cnnl_contiguous(bottom_grad, memory_format);

  // get ptr of tensors
  auto top_grad_impl = torch_mlu::getMluTensorImpl(top_grad_);
  auto top_grad_ptr = top_grad_impl->cnnlMalloc();
  auto best_bboxes_impl = torch_mlu::getMluTensorImpl(best_bboxes_contiguous);
  auto best_bboxes_ptr = best_bboxes_impl->cnnlMalloc();
  auto bottom_grad_impl = torch_mlu::getMluTensorImpl(bottom_grad_);
  auto bottom_grad_ptr = bottom_grad_impl->cnnlMalloc();

  MluOpTensorDescriptor top_grad_desc, best_bboxes_desc, bottom_grad_desc;
  top_grad_desc.set_with_layout(top_grad_, MLUOP_LAYOUT_NHWC);
  best_bboxes_desc.set(best_bboxes_contiguous);
  bottom_grad_desc.set_with_layout(bottom_grad_, MLUOP_LAYOUT_NHWC);

  // get compute handle
  auto handle = mluOpGetCurrentHandle();
  TORCH_MLUOP_CHECK(mluOpRotatedFeatureAlignBackward(
      handle, top_grad_desc.desc(), top_grad_ptr, best_bboxes_desc.desc(),
      best_bboxes_ptr, spatial_scale, points, bottom_grad_desc.desc(),
      bottom_grad_ptr));
  bottom_grad.copy_(bottom_grad_);
}

void rotated_feature_align_forward_mlu(const Tensor features,
                                       const Tensor best_bboxes,
                                       const float spatial_scale,
                                       const int points, Tensor output) {
  RotatedFeatureAlignForwardMLUKernelLauncher(features, best_bboxes,
                                              spatial_scale, points, output);
}

void rotated_feature_align_backward_mlu(const Tensor top_grad,
                                        const Tensor best_bboxes,
                                        const float spatial_scale,
                                        const int points, Tensor bottom_grad) {
  RotatedFeatureAlignBackwardMLUKernelLauncher(
      top_grad, best_bboxes, spatial_scale, points, bottom_grad);
}

void rotated_feature_align_forward_impl(const Tensor features,
                                        const Tensor best_bboxes,
                                        const float spatial_scale,
                                        const int points, Tensor output);

void rotated_feature_align_backward_impl(const Tensor top_grad,
                                         const Tensor best_bboxes,
                                         const float spatial_scale,
                                         const int points, Tensor bottom_grad);

REGISTER_DEVICE_IMPL(rotated_feature_align_forward_impl, MLU,
                     rotated_feature_align_forward_mlu);
REGISTER_DEVICE_IMPL(rotated_feature_align_backward_impl, MLU,
                     rotated_feature_align_backward_mlu);
