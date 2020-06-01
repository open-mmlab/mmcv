#include "pytorch_cpp_helper.hpp"

int DeformRoIPoolForwardCUDALaucher(Tensor input, Tensor rois, Tensor offset,
        Tensor output, int pooled_height, int pooled_width, float spatial_scale,
        int sampling_ratio, float gamma);

int DeformRoIPoolBackwardCUDALaucher(Tensor grad_output, Tensor input, Tensor rois,
        Tensor offset, Tensor grad_input, Tensor grad_offset, int pooled_height,
        int pooled_width, float spatial_scale, int sampling_ratio, float gamma);

int deform_roi_pool_forward(
    Tensor input,
    Tensor rois,
    Tensor offset,
    Tensor output,
    int pooled_height,
    int pooled_width,
    float spatial_scale,
    int sampling_ratio,
    float gamma) {

    if (input.device().is_cuda()) {
        CHECK_CUDA_INPUT(input);
        CHECK_CUDA_INPUT(rois);
        CHECK_CUDA_INPUT(offset);
        CHECK_CUDA_INPUT(output);

        DeformRoIPoolForwardCUDALaucher(input, rois, offset, output,
            pooled_height, pooled_width, spatial_scale, sampling_ratio,
            gamma);
    }

    return 0;
}

int deform_roi_pool_backward(
    Tensor grad_output,
    Tensor input,
    Tensor rois,
    Tensor offset,
    Tensor grad_input,
    Tensor grad_offset,
    int pooled_height,
    int pooled_width,
    float spatial_scale,
    int sampling_ratio,
    float gamma) {

    if (grad_output.device().is_cuda()) {
        CHECK_CUDA_INPUT(grad_output);
        CHECK_CUDA_INPUT(input);
        CHECK_CUDA_INPUT(rois);
        CHECK_CUDA_INPUT(offset);
        CHECK_CUDA_INPUT(grad_input);
        CHECK_CUDA_INPUT(grad_offset);

        DeformRoIPoolBackwardCUDALaucher(grad_output, input, rois, offset,
            grad_input, grad_offset, pooled_height, pooled_width, spatial_scale,
            sampling_ratio, gamma);
    }

    return 0;
}
