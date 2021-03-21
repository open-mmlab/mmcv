// Modified from
// from https://github.com/rosinality/stylegan2-pytorch/blob/master/op/fused_bias_act.cpp
#include "pytorch_cpp_helper.hpp"


torch::Tensor fused_bias_leakyrelu_op(const torch::Tensor& input, const torch::Tensor& bias, const torch::Tensor& refer,
    int act, int grad, float alpha, float scale);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor fused_bias_leakyrelu(const torch::Tensor& input, const torch::Tensor& bias, const torch::Tensor& refer,
    int act, int grad, float alpha, float scale) {
    CHECK_CUDA(input);
    CHECK_CUDA(bias);

    return fused_bias_leakyrelu_op(input, bias, refer, act, grad, alpha, scale);
}

