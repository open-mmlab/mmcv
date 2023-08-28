/*************************************************************************
 * Copyright (C) 2021 Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include <string>
#include <vector>

#include "mlu_common_helper.h"

void sigmoid_focal_loss_forward_mlu(Tensor input, Tensor target, Tensor weight,
                                    Tensor output, const float gamma,
                                    const float alpha) {
  // params check
  TORCH_CHECK(gamma >= 0, "gamma should be greater than or equal to 0. ",
              "But now gamma is ", gamma, ".");

  // check dtype
  TORCH_CHECK(
      input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf,
      "Data type of input should be Float or Half. But now input type is ",
      input.scalar_type(), ".");

  TORCH_CHECK(
      (target.scalar_type() == at::kInt || target.scalar_type() == at::kLong),
      "target type should be Int or Long. ", "But now target type is ",
      target.scalar_type(), ".");

  if (weight.data_ptr() != nullptr) {
    TORCH_CHECK(weight.scalar_type() == input.scalar_type(),
                "Data types of input and weight should be the same. But now "
                "input type is ",
                input.scalar_type(), ", weight type is ", weight.scalar_type(),
                ".");
  } else {
    CNLOG(INFO) << "weight is a empty tensor.";
  }

  // return if zero-element
  if (input.numel() == 0 || target.numel() == 0 || output.numel() == 0) {
    return;
  }

  // contiguous
  auto input_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      input, input.suggest_memory_format());
  // target only support in32
  auto target_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      target.toType(at::kInt), target.suggest_memory_format());
  auto weight_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      weight, weight.suggest_memory_format());
  auto output_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      output, output.suggest_memory_format());

  // set tensor descriptor
  MluOpTensorDescriptor input_desc, target_desc, weight_desc, output_desc;
  input_desc.set(input_contiguous);
  target_desc.set(target_contiguous);
  weight_desc.set(weight_contiguous);
  output_desc.set(output_contiguous);

  // get ptr of tensors
  auto input_impl = torch_mlu::getMluTensorImpl(input_contiguous);
  auto input_ptr = input_impl->cnnlMalloc();
  auto target_impl = torch_mlu::getMluTensorImpl(target_contiguous);
  auto target_ptr = target_impl->cnnlMalloc();
  auto weight_impl = torch_mlu::getMluTensorImpl(weight_contiguous);
  auto weight_ptr = weight_impl->cnnlMalloc();
  auto output_impl = torch_mlu::getMluTensorImpl(output_contiguous);
  auto output_ptr = output_impl->cnnlMalloc();

  // set prefer computation performance and redcuntion approach
  mluOpComputationPreference_t prefer = MLUOP_COMPUTATION_FAST;
  mluOpLossReduction_t reduction = MLUOP_LOSS_REDUCTION_NONE;

  auto handle = mluOpGetCurrentHandle();

  // launch kernel
  TORCH_MLUOP_CHECK(mluOpFocalLossSigmoidForward(
      handle, prefer, reduction, input_desc.desc(), input_ptr,
      target_desc.desc(), target_ptr, weight_desc.desc(), weight_ptr, alpha,
      gamma, output_desc.desc(), output_ptr));
}

void sigmoid_focal_loss_backward_mlu(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, const float gamma,
                                     const float alpha) {
  // params check
  TORCH_CHECK(gamma >= 0, "gamma should be greater than or equal to 0. ",
              "But now gamma is ", gamma, ".");
  // check dtype
  TORCH_CHECK(
      input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf,
      "Data type of input should be Float or Half. But now input type is ",
      input.scalar_type(), ".");

  TORCH_CHECK(
      (target.scalar_type() == at::kInt || target.scalar_type() == at::kLong),
      "target type should be Int or Long. ", "But now target type is ",
      target.scalar_type(), ".");

  bool has_weight = false;
  if (weight.data_ptr() != nullptr) {
    TORCH_CHECK(weight.scalar_type() == input.scalar_type(),
                "Data types of input and weight should be the same. But now "
                "input type is ",
                input.scalar_type(), ", weight type is ", weight.scalar_type(),
                ".");
    has_weight = true;
  } else {
    CNLOG(INFO) << "weight is a empty tensor.";
  }

  if (input.numel() == 0 || target.numel() == 0 || output.numel() == 0) {
    // return if zero-element
    return;
  }

  // contiguous
  auto input_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      input, input.suggest_memory_format());
  // only support in32
  auto target_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      target.toType(at::kInt), target.suggest_memory_format());
  auto weight_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      weight, weight.suggest_memory_format());
  auto output_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      output, output.suggest_memory_format());

  // set tensor descriptor
  MluOpTensorDescriptor input_desc, target_desc, weight_desc, output_desc;
  input_desc.set(input_contiguous);
  target_desc.set(target_contiguous);
  weight_desc.set(weight_contiguous);
  output_desc.set(output_contiguous);

  // get ptr of tensors
  auto input_impl = torch_mlu::getMluTensorImpl(input_contiguous);
  auto input_ptr = input_impl->cnnlMalloc();
  auto target_impl = torch_mlu::getMluTensorImpl(target_contiguous);
  auto target_ptr = target_impl->cnnlMalloc();
  auto weight_impl = torch_mlu::getMluTensorImpl(weight_contiguous);
  auto weight_ptr = weight_impl->cnnlMalloc();
  auto output_impl = torch_mlu::getMluTensorImpl(output_contiguous);
  auto output_ptr = output_impl->cnnlMalloc();

  // set prefer computation performance and redcuntion approach
  // backward only support MLUOP_COMPUTATION_HIGH_PRECISION
  mluOpComputationPreference_t prefer = MLUOP_COMPUTATION_HIGH_PRECISION;
  mluOpLossReduction_t reduction = MLUOP_LOSS_REDUCTION_NONE;

  auto handle = mluOpGetCurrentHandle();

  // launch kernel
  TORCH_MLUOP_CHECK(mluOpFocalLossSigmoidBackward(
      handle, prefer, reduction, input_desc.desc(), input_ptr,
      target_desc.desc(), target_ptr, weight_desc.desc(), weight_ptr, alpha,
      gamma, output_desc.desc(), output_ptr));
}

void sigmoid_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha);

void sigmoid_focal_loss_backward_impl(Tensor input, Tensor target,
                                      Tensor weight, Tensor grad_input,
                                      float gamma, float alpha);

REGISTER_DEVICE_IMPL(sigmoid_focal_loss_forward_impl, MLU,
                     sigmoid_focal_loss_forward_mlu);
REGISTER_DEVICE_IMPL(sigmoid_focal_loss_backward_impl, MLU,
                     sigmoid_focal_loss_backward_mlu);
