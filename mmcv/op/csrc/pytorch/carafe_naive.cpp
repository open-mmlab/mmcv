#include "pytorch_cpp_helper.hpp"

int CARAFENAIVEForwardCUDAKernelLauncher(
    const Tensor features, const Tensor masks, Tensor output,
    const int kernel_size, const int group_size, const int scale_factor);

int CARAFENAIVEBackwardCUDAKernelLauncher(
    const Tensor top_grad, const Tensor features, const Tensor masks,
    Tensor bottom_grad, Tensor mask_grad,
    const int kernel_size, const int group_size, const int scale_factor);

int carafe_naive_forward_cuda(
    Tensor features, Tensor masks, Tensor output,
    int kernel_size, int group_size, int scale_factor) {

    return CARAFENAIVEForwardCUDAKernelLauncher(
               features, masks, output, kernel_size, group_size, scale_factor);
}

int carafe_naive_backward_cuda(
    Tensor top_grad, Tensor features, Tensor masks,
    Tensor bottom_grad, Tensor mask_grad,
    int kernel_size, int group_size, int scale_factor) {

    return CARAFENAIVEBackwardCUDAKernelLauncher(
               top_grad, features, masks, bottom_grad, mask_grad,
               kernel_size, group_size, scale_factor);
}

int carafe_naive_forward(
    Tensor features, Tensor masks, Tensor output,
    int kernel_size, int group_size, int scale_factor) {

    if (features.device().is_cuda()) {
        CHECK_CUDA_INPUT(features);
        CHECK_CUDA_INPUT(masks);
        CHECK_CUDA_INPUT(output);
        return carafe_naive_forward_cuda(
                   features, masks, output,
                   kernel_size, group_size, scale_factor);
    }
    return 0;
}

int carafe_naive_backward(
    Tensor top_grad, Tensor features, Tensor masks,
    Tensor bottom_grad, Tensor mask_grad,
    int kernel_size, int group_size, int scale_factor) {

    if (top_grad.device().is_cuda()) {
        CHECK_CUDA_INPUT(top_grad);
        CHECK_CUDA_INPUT(features);
        CHECK_CUDA_INPUT(masks);
        CHECK_CUDA_INPUT(bottom_grad);
        CHECK_CUDA_INPUT(mask_grad);
        return carafe_naive_backward_cuda(
                   top_grad, features, masks, bottom_grad, mask_grad,
                   kernel_size, group_size, scale_factor);
    }
    return 0;
}
