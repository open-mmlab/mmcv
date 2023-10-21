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

void TINShiftForwardMLUKernelLauncher(Tensor input, Tensor shift,
                                      Tensor output) {
  // params check
  TORCH_CHECK(
      input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf,
      "input type should be Float or Half, got ", input.scalar_type(), ".");
  TORCH_CHECK(input.dim() == 4, "input should be a 4d tensor, got ",
              input.dim(), "d.");
  TORCH_CHECK(shift.dim() == 2, "shift should be a 2d tensor, got ",
              shift.dim(), "d.");
  TORCH_CHECK(
      input.size(0) == shift.size(0),
      "input batch size should be the same as shift's, input batch size is ",
      input.size(0), " and shift batch size is ", shift.size(0), ".");
  TORCH_CHECK(input.size(0) != 0, "Input batch size should not be zero.");
  TORCH_CHECK(input.size(3) != 0,
              "The last dim size of input should not be zero.");
  if (input.size(1) == 0) {
    return;
  }

  // set contiguous
  auto input_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      input, input.suggest_memory_format());
  auto shift_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      shift, shift.suggest_memory_format());
  auto output_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      output, output.suggest_memory_format());

  // get tensor impl
  auto input_impl = torch_mlu::getMluTensorImpl(input_contiguous);
  auto shift_impl = torch_mlu::getMluTensorImpl(shift_contiguous);
  auto output_impl = torch_mlu::getMluTensorImpl(output_contiguous);

  // get the mlu ptr
  auto input_ptr = input_impl->cnnlMalloc();
  auto shift_ptr = shift_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  // set tensor descriptor
  MluOpTensorDescriptor input_desc, shift_desc, output_desc;
  input_desc.set(input_contiguous);
  shift_desc.set(shift_contiguous);
  output_desc.set(output_contiguous);

  // get current handle
  auto handle = mluOpGetCurrentHandle();

  TORCH_MLUOP_CHECK(mluOpTinShiftForward(handle, input_desc.desc(), input_ptr,
                                         shift_desc.desc(), shift_ptr,
                                         output_desc.desc(), output_ptr));
}

void TINShiftBackwardMLUKernelLauncher(Tensor grad_output, Tensor shift,
                                       Tensor grad_input) {
  // params check
  TORCH_CHECK(grad_output.scalar_type() == at::kFloat ||
                  grad_output.scalar_type() == at::kHalf,
              "grad_output type should be Float or Half, got ",
              grad_output.scalar_type(), ".");
  TORCH_CHECK(grad_output.dim() == 4, "grad_output should be a 4d tensor, got ",
              grad_output.dim(), "d.");
  TORCH_CHECK(shift.dim() == 2, "shift should be a 2d tensor, got ",
              shift.dim(), "d.");
  TORCH_CHECK(grad_output.size(0) == shift.size(0),
              "grad_output batch size should be the same as shift's, "
              "grad_output batch size is ",
              grad_output.size(0), ", shift batch size is ", shift.size(0),
              ".");
  TORCH_CHECK(grad_output.size(0) != 0,
              "grad_output batch size should not be zero.");
  TORCH_CHECK(grad_output.size(3) != 0,
              "The last dim size of grad_output should not be zero.");
  if (grad_output.size(1) == 0) {
    return;
  }

  // set contiguous
  auto grad_output_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      grad_output, grad_output.suggest_memory_format());
  auto shift_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      shift, shift.suggest_memory_format());
  auto grad_input_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      grad_input, grad_input.suggest_memory_format());

  // get tensor impl
  auto grad_output_impl = torch_mlu::getMluTensorImpl(grad_output_contiguous);
  auto shift_impl = torch_mlu::getMluTensorImpl(shift_contiguous);
  auto grad_input_impl = torch_mlu::getMluTensorImpl(grad_input_contiguous);

  // get the mlu ptr
  auto grad_output_ptr = grad_output_impl->cnnlMalloc();
  auto shift_ptr = shift_impl->cnnlMalloc();
  auto grad_input_ptr = grad_input_impl->cnnlMalloc();

  // set tensor descriptor
  MluOpTensorDescriptor grad_output_desc, shift_desc, grad_input_desc;
  grad_output_desc.set(grad_output_contiguous);
  shift_desc.set(shift_contiguous);
  grad_input_desc.set(grad_input_contiguous);

  // get current handle
  auto handle = mluOpGetCurrentHandle();

  TORCH_MLUOP_CHECK(mluOpTinShiftBackward(
      handle, grad_output_desc.desc(), grad_output_ptr, shift_desc.desc(),
      shift_ptr, grad_input_desc.desc(), grad_input_ptr));
}

void tin_shift_forward_mlu(Tensor input, Tensor shift, Tensor output) {
  TINShiftForwardMLUKernelLauncher(input, shift, output);
}

void tin_shift_backward_mlu(Tensor grad_output, Tensor shift,
                            Tensor grad_input) {
  TINShiftBackwardMLUKernelLauncher(grad_output, shift, grad_input);
}

void tin_shift_forward_impl(Tensor input, Tensor shift, Tensor output);

void tin_shift_backward_impl(Tensor grad_output, Tensor shift,
                             Tensor grad_input);

REGISTER_DEVICE_IMPL(tin_shift_forward_impl, MLU, tin_shift_forward_mlu);
REGISTER_DEVICE_IMPL(tin_shift_backward_impl, MLU, tin_shift_backward_mlu);
