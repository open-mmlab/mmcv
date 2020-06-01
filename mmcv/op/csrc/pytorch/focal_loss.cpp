#include "pytorch_cpp_helper.hpp"

int SigmoidFocalLossForwardCUDALaucher(Tensor input, Tensor target, Tensor weight, Tensor output,
                                       const float gamma, const float alpha);

int SigmoidFocalLossBackwardCUDALaucher(Tensor input, Tensor target, Tensor weight, Tensor grad_input,
                                       const float gamma, const float alpha);

int SoftmaxFocalLossForwardCUDALaucher(Tensor input, Tensor target, Tensor weight, Tensor output,
                                       const float gamma, const float alpha);

int SoftmaxFocalLossBackwardCUDALaucher(Tensor input, Tensor target, Tensor weight, Tensor buff,
                                        Tensor grad_input, const float gamma, const float alpha);

int sigmoid_focal_loss_forward(
    Tensor input,
    Tensor target,
    Tensor weight,
    Tensor output,
    float gamma, 
    float alpha) {

    if(input.device().is_cuda()){
        CHECK_CUDA_INPUT(input);
        CHECK_CUDA_INPUT(target);
        CHECK_CUDA_INPUT(weight);
        CHECK_CUDA_INPUT(output);

        SigmoidFocalLossForwardCUDALaucher(input, target, weight, output, gamma, alpha);
    }
    return 0;
}

int sigmoid_focal_loss_backward(
    Tensor input,
    Tensor target,
    Tensor weight,
    Tensor grad_input,
    float gamma, 
    float alpha) {

    if(input.device().is_cuda()){
        CHECK_CUDA_INPUT(input);
        CHECK_CUDA_INPUT(target);
        CHECK_CUDA_INPUT(weight);
        CHECK_CUDA_INPUT(grad_input);

        SigmoidFocalLossBackwardCUDALaucher(input, target, weight, grad_input, gamma, alpha);
    }
    return 0;
}

int softmax_focal_loss_forward(
    Tensor input,
    Tensor target,
    Tensor weight,
    Tensor output,
    float gamma, 
    float alpha) {

    if(input.device().is_cuda()){
        CHECK_CUDA_INPUT(input);
        CHECK_CUDA_INPUT(target);
        CHECK_CUDA_INPUT(weight);
        CHECK_CUDA_INPUT(output);

        SoftmaxFocalLossForwardCUDALaucher(input, target, weight, output, gamma, alpha);
    }
    return 0;
}

int softmax_focal_loss_backward(
    Tensor input,
    Tensor target,
    Tensor weight,
    Tensor buff,
    Tensor grad_input,
    float gamma, 
    float alpha) {

    if(input.device().is_cuda()){
        CHECK_CUDA_INPUT(input);
        CHECK_CUDA_INPUT(target);
        CHECK_CUDA_INPUT(weight);
        CHECK_CUDA_INPUT(buff);
        CHECK_CUDA_INPUT(grad_input);

        SoftmaxFocalLossBackwardCUDALaucher(input, target, weight, buff, grad_input, gamma, alpha);
    }
    return 0;
}

