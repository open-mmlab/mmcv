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

void PSAMaskForwardMLUKernelLauncher(const int psa_type, const Tensor x,
                                     Tensor y, const int num_,
                                     const int h_feature, const int w_feature,
                                     const int h_mask, const int w_mask,
                                     const int half_h_mask,
                                     const int half_w_mask) {
  int y_c = y.size(1);

  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(x.dim());
  auto x_tensor = torch_mlu::cnnl::ops::cnnl_contiguous(x, memory_format);
  at::Tensor y_tmp =
      at::empty({num_, y_c, h_feature, w_feature}, x.options(), memory_format);

  MluOpTensorDescriptor x_desc, y_desc;
  x_desc.set_with_layout(x_tensor, MLUOP_LAYOUT_NHWC);
  y_desc.set_with_layout(y_tmp, MLUOP_LAYOUT_NHWC);

  auto handle = mluOpGetCurrentHandle();
  auto x_impl = torch_mlu::getMluTensorImpl(x_tensor);
  auto x_ptr = x_impl->cnnlMalloc();
  auto y_impl = torch_mlu::getMluTensorImpl(y_tmp);
  auto y_ptr = y_impl->cnnlMalloc();

  TORCH_MLUOP_CHECK(mluOpPsamaskForward(handle, psa_type, x_desc.desc(), x_ptr,
                                        h_mask, w_mask, y_desc.desc(), y_ptr));

  y.copy_(y_tmp);
}

void PSAMaskBackwardMLUKernelLauncher(const int psa_type, const Tensor dy,
                                      Tensor dx, const int num_,
                                      const int h_feature, const int w_feature,
                                      const int h_mask, const int w_mask,
                                      const int half_h_mask,
                                      const int half_w_mask) {
  int dx_c = dx.size(1);

  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(dy.dim());
  auto dy_tensor = torch_mlu::cnnl::ops::cnnl_contiguous(dy, memory_format);
  at::Tensor dx_tmp = at::empty({num_, dx_c, h_feature, w_feature},
                                dy.options(), memory_format);

  MluOpTensorDescriptor dy_desc, dx_tmp_desc;
  dy_desc.set_with_layout(dy_tensor, MLUOP_LAYOUT_NHWC);
  dx_tmp_desc.set_with_layout(dx_tmp, MLUOP_LAYOUT_NHWC);

  auto handle = mluOpGetCurrentHandle();

  // get ptr of tensors
  auto dx_impl = torch_mlu::getMluTensorImpl(dx_tmp);
  auto dx_ptr = dx_impl->cnnlMalloc();
  auto dy_impl = torch_mlu::getMluTensorImpl(dy_tensor);
  auto dy_ptr = dy_impl->cnnlMalloc();

  TORCH_MLUOP_CHECK(mluOpPsamaskBackward(handle, psa_type, dy_desc.desc(),
                                         dy_ptr, h_mask, w_mask,
                                         dx_tmp_desc.desc(), dx_ptr));

  dx.copy_(dx_tmp);
}

void psamask_forward_mlu(const int psa_type, const Tensor input, Tensor output,
                         const int num_, const int h_feature,
                         const int w_feature, const int h_mask,
                         const int w_mask, const int half_h_mask,
                         const int half_w_mask) {
  PSAMaskForwardMLUKernelLauncher(psa_type, input, output, num_, h_feature,
                                  w_feature, h_mask, w_mask, half_h_mask,
                                  half_w_mask);
}

void psamask_backward_mlu(const int psa_type, const Tensor grad_output,
                          Tensor grad_input, const int num_,
                          const int h_feature, const int w_feature,
                          const int h_mask, const int w_mask,
                          const int half_h_mask, const int half_w_mask) {
  PSAMaskBackwardMLUKernelLauncher(psa_type, grad_output, grad_input, num_,
                                   h_feature, w_feature, h_mask, w_mask,
                                   half_h_mask, half_w_mask);
}

void psamask_forward_impl(const int psa_type, const Tensor input, Tensor output,
                          const int num_, const int h_feature,
                          const int w_feature, const int h_mask,
                          const int w_mask, const int half_h_mask,
                          const int half_w_mask);

void psamask_backward_impl(const int psa_type, const Tensor grad_output,
                           Tensor grad_input, const int num_,
                           const int h_feature, const int w_feature,
                           const int h_mask, const int w_mask,
                           const int half_h_mask, const int half_w_mask);

REGISTER_DEVICE_IMPL(psamask_forward_impl, MLU, psamask_forward_mlu);
REGISTER_DEVICE_IMPL(psamask_backward_impl, MLU, psamask_backward_mlu);
