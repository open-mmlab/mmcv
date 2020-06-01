#include "pytorch_cpp_helper.hpp"

int DeformConvForwardCUDALauncher(
    Tensor input, Tensor weight, Tensor offset, Tensor output,
    Tensor columns, Tensor ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationW, int dilationH, int group,
    int deformable_group, int im2col_step);

int DeformConvBackwardInputCUDALauncher(
    Tensor input, Tensor offset, Tensor gradOutput, Tensor gradInput,
    Tensor gradOffset, Tensor weight, Tensor columns,
    int kW, int kH, int dW, int dH, int padW, int padH,
    int dilationW, int dilationH, int group, int deformable_group, int im2col_step);

int DeformConvBackwardParametersCUDALauncher(
    Tensor input, Tensor offset, Tensor gradOutput, Tensor gradWeight,
    Tensor columns, Tensor ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationW, int dilationH, int group,
    int deformable_group, float scale, int im2col_step);

int deform_conv_forward(
    Tensor input, Tensor weight, Tensor offset, Tensor output,
    Tensor columns, Tensor ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationW, int dilationH, int group,
    int deformable_group, int im2col_step) {

    if (input.device().is_cuda()) {
        CHECK_CUDA_INPUT(input);
        CHECK_CUDA_INPUT(offset);
        CHECK_CUDA_INPUT(weight);
        CHECK_CUDA_INPUT(output);
        CHECK_CUDA_INPUT(columns);
        CHECK_CUDA_INPUT(ones);

        DeformConvForwardCUDALauncher(
            input, weight, offset, output, columns, ones,
            kW, kH, dW, dH, padW, padH, dilationW, dilationH,
            group, deformable_group, im2col_step);
    }
    return 0;
}

int deform_conv_backward_input(
    Tensor input, Tensor offset, Tensor gradOutput, Tensor gradInput,
    Tensor gradOffset, Tensor weight, Tensor columns,
    int kW, int kH, int dW, int dH, int padW, int padH,
    int dilationW, int dilationH, int group, int deformable_group, int im2col_step) {

    if (input.device().is_cuda()) {
        CHECK_CUDA_INPUT(input);
        CHECK_CUDA_INPUT(offset);
        CHECK_CUDA_INPUT(gradOutput);
        CHECK_CUDA_INPUT(gradInput);
        CHECK_CUDA_INPUT(gradOffset);
        CHECK_CUDA_INPUT(weight);
        CHECK_CUDA_INPUT(columns);

        DeformConvBackwardInputCUDALauncher(
            input, offset, gradOutput, gradInput, gradOffset,
            weight, columns, kW, kH, dW, dH, padW, padH, dilationW, dilationH,
            group, deformable_group, im2col_step);
    }
    return 0;
}

int deform_conv_backward_parameters(
    Tensor input, Tensor offset, Tensor gradOutput, Tensor gradWeight,
    Tensor columns, Tensor ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationW, int dilationH, int group,
    int deformable_group, float scale, int im2col_step) {

    if (input.device().is_cuda()) {
        CHECK_CUDA_INPUT(input);
        CHECK_CUDA_INPUT(offset);
        CHECK_CUDA_INPUT(gradOutput);
        CHECK_CUDA_INPUT(gradWeight);
        CHECK_CUDA_INPUT(columns);
        CHECK_CUDA_INPUT(ones);

        DeformConvBackwardParametersCUDALauncher(
            input, offset, gradOutput, gradWeight, columns, ones,
            kW, kH, dW, dH, padW, padH, dilationW, dilationH,
            group, deformable_group, scale, im2col_step);
    }
    return 0;
}
