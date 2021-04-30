// Modified from
// from
// https://github.com/rosinality/stylegan2-pytorch/blob/master/op/fused_bias_act.cpp
#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
torch::Tensor fused_bias_leakyrelu_op(const torch::Tensor &input,
                                      const torch::Tensor &bias,
                                      const torch::Tensor &refer, int act,
                                      int grad, float alpha, float scale);

#endif

torch::Tensor fused_bias_leakyrelu(const torch::Tensor &input,
                                   const torch::Tensor &bias,
                                   const torch::Tensor &refer, int act,
                                   int grad, float alpha, float scale) {
#ifdef MMCV_WITH_CUDA
  CHECK_CUDA(input);
  CHECK_CUDA(bias);

  return fused_bias_leakyrelu_op(input, bias, refer, act, grad, alpha, scale);
#else
  AT_ERROR("Fused bias leakyrelu is not compiled with GPU support");
#endif
}
