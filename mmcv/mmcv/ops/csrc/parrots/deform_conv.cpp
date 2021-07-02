#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void DeformConvForwardCUDAKernelLauncher(Tensor input, Tensor weight,
                                         Tensor offset, Tensor output,
                                         Tensor columns, Tensor ones, int kW,
                                         int kH, int dW, int dH, int padW,
                                         int padH, int dilationW, int dilationH,
                                         int group, int deformable_group,
                                         int im2col_step);

void DeformConvBackwardInputCUDAKernelLauncher(
    Tensor input, Tensor offset, Tensor gradOutput, Tensor gradInput,
    Tensor gradOffset, Tensor weight, Tensor columns, int kW, int kH, int dW,
    int dH, int padW, int padH, int dilationW, int dilationH, int group,
    int deformable_group, int im2col_step);

void DeformConvBackwardParametersCUDAKernelLauncher(
    Tensor input, Tensor offset, Tensor gradOutput, Tensor gradWeight,
    Tensor columns, Tensor ones, int kW, int kH, int dW, int dH, int padW,
    int padH, int dilationW, int dilationH, int group, int deformable_group,
    float scale, int im2col_step);

void deform_conv_forward_cuda(Tensor input, Tensor weight, Tensor offset,
                              Tensor output, Tensor columns, Tensor ones,
                              int kW, int kH, int dW, int dH, int padW,
                              int padH, int dilationW, int dilationH, int group,
                              int deformable_group, int im2col_step) {
  DeformConvForwardCUDAKernelLauncher(
      input, weight, offset, output, columns, ones, kW, kH, dW, dH, padW, padH,
      dilationW, dilationH, group, deformable_group, im2col_step);
}

void deform_conv_backward_input_cuda(Tensor input, Tensor offset,
                                     Tensor gradOutput, Tensor gradInput,
                                     Tensor gradOffset, Tensor weight,
                                     Tensor columns, int kW, int kH, int dW,
                                     int dH, int padW, int padH, int dilationW,
                                     int dilationH, int group,
                                     int deformable_group, int im2col_step) {
  DeformConvBackwardInputCUDAKernelLauncher(
      input, offset, gradOutput, gradInput, gradOffset, weight, columns, kW, kH,
      dW, dH, padW, padH, dilationW, dilationH, group, deformable_group,
      im2col_step);
}

void deform_conv_backward_parameters_cuda(
    Tensor input, Tensor offset, Tensor gradOutput, Tensor gradWeight,
    Tensor columns, Tensor ones, int kW, int kH, int dW, int dH, int padW,
    int padH, int dilationW, int dilationH, int group, int deformable_group,
    float scale, int im2col_step) {
  DeformConvBackwardParametersCUDAKernelLauncher(
      input, offset, gradOutput, gradWeight, columns, ones, kW, kH, dW, dH,
      padW, padH, dilationW, dilationH, group, deformable_group, scale,
      im2col_step);
}
#endif

void deform_conv_forward(Tensor input, Tensor weight, Tensor offset,
                         Tensor output, Tensor columns, Tensor ones, int kW,
                         int kH, int dW, int dH, int padW, int padH,
                         int dilationW, int dilationH, int group,
                         int deformable_group, int im2col_step) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(offset);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(output);
    CHECK_CUDA_INPUT(columns);
    CHECK_CUDA_INPUT(ones);

    deform_conv_forward_cuda(input, weight, offset, output, columns, ones, kW,
                             kH, dW, dH, padW, padH, dilationW, dilationH,
                             group, deformable_group, im2col_step);
#else
    AT_ERROR("DeformConv is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("DeformConv is not implemented on CPU");
  }
}

void deform_conv_backward_input(Tensor input, Tensor offset, Tensor gradOutput,
                                Tensor gradInput, Tensor gradOffset,
                                Tensor weight, Tensor columns, int kW, int kH,
                                int dW, int dH, int padW, int padH,
                                int dilationW, int dilationH, int group,
                                int deformable_group, int im2col_step) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(offset);
    CHECK_CUDA_INPUT(gradOutput);
    CHECK_CUDA_INPUT(gradInput);
    CHECK_CUDA_INPUT(gradOffset);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(columns);

    deform_conv_backward_input_cuda(input, offset, gradOutput, gradInput,
                                    gradOffset, weight, columns, kW, kH, dW, dH,
                                    padW, padH, dilationW, dilationH, group,
                                    deformable_group, im2col_step);
#else
    AT_ERROR("DeformConv is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("DeformConv is not implemented on CPU");
  }
}

void deform_conv_backward_parameters(Tensor input, Tensor offset,
                                     Tensor gradOutput, Tensor gradWeight,
                                     Tensor columns, Tensor ones, int kW,
                                     int kH, int dW, int dH, int padW, int padH,
                                     int dilationW, int dilationH, int group,
                                     int deformable_group, float scale,
                                     int im2col_step) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(offset);
    CHECK_CUDA_INPUT(gradOutput);
    CHECK_CUDA_INPUT(gradWeight);
    CHECK_CUDA_INPUT(columns);
    CHECK_CUDA_INPUT(ones);

    deform_conv_backward_parameters_cuda(input, offset, gradOutput, gradWeight,
                                         columns, ones, kW, kH, dW, dH, padW,
                                         padH, dilationW, dilationH, group,
                                         deformable_group, scale, im2col_step);
#else
    AT_ERROR("DeformConv is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("DeformConv is not implemented on CPU");
  }
}
