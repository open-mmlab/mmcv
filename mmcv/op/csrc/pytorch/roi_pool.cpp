#include "pytorch_cpp_helper.hpp"

int ROIPoolForwardCUDAKernelLauncher(
    Tensor input, Tensor rois, Tensor output, Tensor argmax,
    int pooled_height, int pooled_width, float spatial_scale);

int ROIPoolBackwardCUDAKernelLauncher(
    Tensor grad_output, Tensor rois, Tensor argmax, Tensor grad_input,
    int pooled_height, int pooled_width, float spatial_scale);

int roi_pool_forward(
    Tensor input,
    Tensor rois,
    Tensor output,
    Tensor argmax,
    int pooled_height,
    int pooled_width,
    float spatial_scale){

    if (input.device().is_cuda()) {
        CHECK_CUDA_INPUT(input);
        CHECK_CUDA_INPUT(rois);
        CHECK_CUDA_INPUT(output);
        CHECK_CUDA_INPUT(argmax);

        ROIPoolForwardCUDAKernelLauncher(
            input, rois, output, argmax,
            pooled_height, pooled_width, spatial_scale);
    }
    return 0;
}

int roi_pool_backward(
    Tensor grad_output,
    Tensor rois,
    Tensor argmax,
    Tensor grad_input,
    int pooled_height, 
    int pooled_width, 
    float spatial_scale){

    if (grad_output.device().is_cuda()) {
        CHECK_CUDA_INPUT(grad_output);
        CHECK_CUDA_INPUT(rois);
        CHECK_CUDA_INPUT(argmax);
        CHECK_CUDA_INPUT(grad_input);

        ROIPoolBackwardCUDAKernelLauncher(
            grad_output, rois, argmax, grad_input,
            pooled_height, pooled_width, spatial_scale);
    }
    return 0;
}
