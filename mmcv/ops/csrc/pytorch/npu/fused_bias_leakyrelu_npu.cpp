#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

Tensor fused_bias_leakyrelu_op_impl(Tensor& input, Tensor& bias,
                                    Tensor& refer, int act, int grad, 
                                    float alpha, float scale) {
}

Tensor fused_bias_leakyrelu_npu(Tensor& input,Tensor& bias,
                                Tensor& refer, int act, int grad, 
                                float alpha, float scale) {
  at::tensor y = at::tempty_like(input);
  // forward
  if (grad == 1){
    auto input_size = input.size();
    int input_length = input_size.size();
    if (input_length > 1){
        for (int i = 0; i < input_length; i++){
            if (i != 1){
                input_size[i] = 1;
            }  
        }
    }
    at::Tensor bias_ = at::reshape(bias, input_size);
    at::Tensor bias_tmp = NPUNativeFunctions::npu_broadcast(bias_, input.size());
    OpCommand cmd;
    cmd.Name("FusedBiasLeakyRelu")
        .Input(input);
        .Input(bias);
        .Output(y);
        .Attr("sacle",sacle);
        .Attr("negative_slope", alpha);
        .Run();
  }

  // backward
  if (grad == 2){
    OpCommand cmd;
    cmd.Name("FusedBiasLeakyReluGrad")
        .Input(input);
        .Input(ref);
        .Output(y);
        .Attr("sacle",sacle);
        .Attr("negative_slope", alpha);
        .Run();
  }
  
  return y;
}

REGISTER_NPU_IMPL(fused_bias_leakyrelu_op_impl,
                  fused_bias_leakyrelu_npu);
