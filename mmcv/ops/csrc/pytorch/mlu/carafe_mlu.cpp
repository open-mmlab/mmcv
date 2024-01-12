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

void CARAFEForwardMLUKernelLauncher(const Tensor input, const Tensor mask,
                                    Tensor rinput, Tensor routput, Tensor rmask,
                                    Tensor output, const int kernel_size,
                                    const int group_size,
                                    const int scale_factor) {
  // check tensor data type
  TORCH_CHECK(
      input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf,
      "Data type of input should be Float or Half. But now input type is ",
      input.scalar_type(), ".");

  TORCH_CHECK(mask.scalar_type() == input.scalar_type(),
              "Data types of input and mask should be the same, but got ",
              input.scalar_type(), " and ", mask.scalar_type());

  // check number of dimensions
  TORCH_CHECK(input.dim() == 4, "input should be a 4-D tensor, but has ",
              input.dim(), "D.");
  TORCH_CHECK(mask.dim() == 4, "mask should be a 4-D tensor, but has ",
              input.dim(), "D.");

  // return fast on zero-element tensor
  if (output.numel() == 0) {
    output = at::zeros(output.sizes().vec(), output.options());
    return;
  }

  // convert NCHW to NHWC
  auto memory_format_input_nhwc =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(input.dim());
  auto rinput_ =
      torch_mlu::cnnl::ops::cnnl_contiguous(input, memory_format_input_nhwc);

  auto memory_format_mask_nhwc =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(mask.dim());
  auto rmask_ =
      torch_mlu::cnnl::ops::cnnl_contiguous(mask, memory_format_mask_nhwc);

  auto memory_format_output_nhwc =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(output.dim());
  auto routput_ =
      torch_mlu::cnnl::ops::cnnl_contiguous(output, memory_format_output_nhwc);

  // set tensor descriptor
  MluOpTensorDescriptor input_desc, mask_desc, output_desc;
  input_desc.set_with_layout(rinput_, MLUOP_LAYOUT_NHWC);
  mask_desc.set_with_layout(rmask_, MLUOP_LAYOUT_NHWC);
  output_desc.set_with_layout(routput_, MLUOP_LAYOUT_NHWC);

  // get ptr of tensors
  auto input_impl = torch_mlu::getMluTensorImpl(rinput_);
  auto input_ptr = input_impl->cnnlMalloc();
  auto mask_impl = torch_mlu::getMluTensorImpl(rmask_);
  auto mask_ptr = mask_impl->cnnlMalloc();
  auto output_impl = torch_mlu::getMluTensorImpl(routput_);
  auto output_ptr = output_impl->cnnlMalloc();

  // set op descriptor
  auto handle = mluOpGetCurrentHandle();
  mluOpCarafeDescriptor_t carafe_desc;
  TORCH_MLUOP_CHECK(mluOpCreateCarafeDescriptor(&carafe_desc));
  TORCH_MLUOP_CHECK(mluOpSetCarafeDescriptor(
      carafe_desc, input.dim(), kernel_size, group_size, scale_factor));
  // launch kernel
  TORCH_MLUOP_CHECK(mluOpCarafeForward(handle, carafe_desc, input_desc.desc(),
                                       input_ptr, mask_desc.desc(), mask_ptr,
                                       output_desc.desc(), output_ptr));
  // destroy op descriptor
  TORCH_MLUOP_CHECK(mluOpDestroyCarafeDescriptor(carafe_desc));

  // copy output from NHWC back into NCHW
  rinput.copy_(rinput_);
  output.copy_(routput_);
}

void CARAFEBackwardMLUKernelLauncher(
    const Tensor grad_output, const Tensor rinput, const Tensor mask,
    Tensor rgrad_output, Tensor rgrad_input_hs, Tensor rgrad_input,
    Tensor rgrad_mask, Tensor grad_input, Tensor grad_mask,
    const int kernel_size, const int group_size, const int scale_factor) {
  // data type check
  TORCH_CHECK(grad_output.scalar_type() == at::kFloat ||
                  grad_output.scalar_type() == at::kHalf,
              "grad_output type should be Float or Half, got ",
              grad_output.scalar_type());
  TORCH_CHECK(grad_output.scalar_type() == mask.scalar_type(),
              "mask should have the same type as grad_output");

  // dim check
  TORCH_CHECK(grad_output.dim() == 4, "grad_output should be a 4d tensor, got ",
              grad_output.dim(), "D");

  // param check
  TORCH_CHECK(kernel_size < 137, "kernel_size should be less than 137, got ",
              kernel_size);

  // convert NCHW to NHWC
  auto memory_format_input_nhwc =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(rinput.dim());
  auto rinput_ =
      torch_mlu::cnnl::ops::cnnl_contiguous(rinput, memory_format_input_nhwc);

  auto memory_format_mask_nhwc =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(mask.dim());
  auto rmask_ =
      torch_mlu::cnnl::ops::cnnl_contiguous(mask, memory_format_mask_nhwc);

  auto memory_format_grad_output_nhwc =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(grad_output.dim());
  auto rgrad_output_ = torch_mlu::cnnl::ops::cnnl_contiguous(
      grad_output, memory_format_grad_output_nhwc);

  auto memory_format_grad_input_nhwc =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(grad_input.dim());
  auto rgrad_input_ = torch_mlu::cnnl::ops::cnnl_contiguous(
                          grad_input, memory_format_grad_input_nhwc)
                          .zero_();

  auto memory_format_grad_mask_nhwc =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(grad_mask.dim());
  auto rgrad_mask_ = torch_mlu::cnnl::ops::cnnl_contiguous(
      grad_mask, memory_format_grad_mask_nhwc);

  // set tensor descriptor
  MluOpTensorDescriptor input_desc, mask_desc;
  input_desc.set_with_layout(rinput_, MLUOP_LAYOUT_NHWC);
  mask_desc.set_with_layout(rmask_, MLUOP_LAYOUT_NHWC);

  MluOpTensorDescriptor grad_output_desc, grad_input_desc, grad_mask_desc;
  grad_output_desc.set_with_layout(rgrad_output_, MLUOP_LAYOUT_NHWC);
  grad_input_desc.set_with_layout(rgrad_input_, MLUOP_LAYOUT_NHWC);
  grad_mask_desc.set_with_layout(rgrad_mask_, MLUOP_LAYOUT_NHWC);

  // get ptr of tensors
  auto input_impl = torch_mlu::getMluTensorImpl(rinput_);
  auto input_ptr = input_impl->cnnlMalloc();
  auto mask_impl = torch_mlu::getMluTensorImpl(rmask_);
  auto mask_ptr = mask_impl->cnnlMalloc();
  auto grad_output_impl = torch_mlu::getMluTensorImpl(rgrad_output_);
  auto grad_output_ptr = grad_output_impl->cnnlMalloc();
  auto grad_input_impl = torch_mlu::getMluTensorImpl(rgrad_input_);
  auto grad_input_ptr = grad_input_impl->cnnlMalloc();
  auto grad_mask_impl = torch_mlu::getMluTensorImpl(rgrad_mask_);
  auto grad_mask_ptr = grad_mask_impl->cnnlMalloc();

  // set op descriptor
  auto handle = mluOpGetCurrentHandle();
  mluOpCarafeDescriptor_t carafe_desc;
  TORCH_MLUOP_CHECK(mluOpCreateCarafeDescriptor(&carafe_desc));
  TORCH_MLUOP_CHECK(mluOpSetCarafeDescriptor(
      carafe_desc, grad_output.dim(), kernel_size, group_size, scale_factor));
  // launch kernel
  TORCH_MLUOP_CHECK(mluOpCarafeBackward(
      handle, carafe_desc, input_desc.desc(), input_ptr, mask_desc.desc(),
      mask_ptr, grad_output_desc.desc(), grad_output_ptr,
      grad_input_desc.desc(), grad_input_ptr, grad_mask_desc.desc(),
      grad_mask_ptr));
  // destroy op descriptor
  TORCH_MLUOP_CHECK(mluOpDestroyCarafeDescriptor(carafe_desc));

  // copy output from NHWC back into NCHW
  grad_input.copy_(rgrad_input_);
  grad_mask.copy_(rgrad_mask_);
}

void carafe_forward_mlu(Tensor features, Tensor masks, Tensor rfeatures,
                        Tensor routput, Tensor rmasks, Tensor output,
                        int kernel_size, int group_size, int scale_factor) {
  CARAFEForwardMLUKernelLauncher(features, masks, rfeatures, routput, rmasks,
                                 output, kernel_size, group_size, scale_factor);
}

void carafe_backward_mlu(Tensor top_grad, Tensor rfeatures, Tensor masks,
                         Tensor rtop_grad, Tensor rbottom_grad_hs,
                         Tensor rbottom_grad, Tensor rmask_grad,
                         Tensor bottom_grad, Tensor mask_grad, int kernel_size,
                         int group_size, int scale_factor) {
  CARAFEBackwardMLUKernelLauncher(top_grad, rfeatures, masks, rtop_grad,
                                  rbottom_grad_hs, rbottom_grad, rmask_grad,
                                  bottom_grad, mask_grad, kernel_size,
                                  group_size, scale_factor);
}

void carafe_forward_impl(Tensor features, Tensor masks, Tensor rfeatures,
                         Tensor routput, Tensor rmasks, Tensor output,
                         int kernel_size, int group_size, int scale_factor);

void carafe_backward_impl(Tensor top_grad, Tensor rfeatures, Tensor masks,
                          Tensor rtop_grad, Tensor rbottom_grad_hs,
                          Tensor rbottom_grad, Tensor rmask_grad,
                          Tensor bottom_grad, Tensor mask_grad, int kernel_size,
                          int group_size, int scale_factor);

REGISTER_DEVICE_IMPL(carafe_forward_impl, MLU, carafe_forward_mlu);
REGISTER_DEVICE_IMPL(carafe_backward_impl, MLU, carafe_backward_mlu);
