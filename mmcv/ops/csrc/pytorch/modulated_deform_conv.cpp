#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void ModulatedDeformConvForwardCUDAKernelLauncher(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor output, Tensor columns, int kernel_h, int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, const int group,
    const int deformable_group, const bool with_bias);

void ModulatedDeformConvBackwardCUDAKernelLauncher(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor columns, Tensor grad_input, Tensor grad_weight,
    Tensor grad_bias, Tensor grad_offset, Tensor grad_mask, Tensor grad_output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
    const bool with_bias);

void modulated_deform_conv_forward_cuda(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor output, Tensor columns, int kernel_h, int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, const int group,
    const int deformable_group, const bool with_bias) {
  ModulatedDeformConvForwardCUDAKernelLauncher(
      input, weight, bias, ones, offset, mask, output, columns, kernel_h,
      kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
      deformable_group, with_bias);
}

void modulated_deform_conv_backward_cuda(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor columns, Tensor grad_input, Tensor grad_weight,
    Tensor grad_bias, Tensor grad_offset, Tensor grad_mask, Tensor grad_output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
    const bool with_bias) {
  ModulatedDeformConvBackwardCUDAKernelLauncher(
      input, weight, bias, ones, offset, mask, columns, grad_input, grad_weight,
      grad_bias, grad_offset, grad_mask, grad_output, kernel_h, kernel_w,
      stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
      deformable_group, with_bias);
}
#endif

void modulated_deform_conv_forward(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor output, Tensor columns, int kernel_h, int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, const int group,
    const int deformable_group, const bool with_bias) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(bias);
    CHECK_CUDA_INPUT(ones);
    CHECK_CUDA_INPUT(offset);
    CHECK_CUDA_INPUT(mask);
    CHECK_CUDA_INPUT(output);
    CHECK_CUDA_INPUT(columns);

    modulated_deform_conv_forward_cuda(
        input, weight, bias, ones, offset, mask, output, columns, kernel_h,
        kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
        group, deformable_group, with_bias);
#else
    AT_ERROR("ModulatedDeformConv is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("ModulatedDeformConv is not implemented on CPU");
  }
}

void modulated_deform_conv_backward(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor columns, Tensor grad_input, Tensor grad_weight,
    Tensor grad_bias, Tensor grad_offset, Tensor grad_mask, Tensor grad_output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
    const bool with_bias) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(bias);
    CHECK_CUDA_INPUT(ones);
    CHECK_CUDA_INPUT(offset);
    CHECK_CUDA_INPUT(mask);
    CHECK_CUDA_INPUT(columns);
    CHECK_CUDA_INPUT(grad_input);
    CHECK_CUDA_INPUT(grad_weight);
    CHECK_CUDA_INPUT(grad_bias);
    CHECK_CUDA_INPUT(grad_offset);
    CHECK_CUDA_INPUT(grad_mask);
    CHECK_CUDA_INPUT(grad_output);

    modulated_deform_conv_backward_cuda(
        input, weight, bias, ones, offset, mask, columns, grad_input,
        grad_weight, grad_bias, grad_offset, grad_mask, grad_output, kernel_h,
        kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
        group, deformable_group, with_bias);
#else
    AT_ERROR("ModulatedDeformConv is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("ModulatedDeformConv is not implemented on CPU");
  }
}
