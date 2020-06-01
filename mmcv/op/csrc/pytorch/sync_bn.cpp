#include "pytorch_cpp_helper.hpp"
#include <cmath>
#include <vector>

using namespace at;

void cudaSyncBNForwardStep1(size_t n, size_t c, size_t h, size_t w,
                            const at::Tensor input, at::Tensor mean);

void cudaSyncBNForwardStep2(size_t n, size_t c, size_t h, size_t w, const at::Tensor input,
                            const at::Tensor mean, at::Tensor var);

void cudaSyncBNForwardStep3(size_t n, size_t c, size_t h, size_t w, size_t group_size, const at::Tensor input,
                            const float eps, const float momentum, const at::Tensor mean, const at::Tensor var,
                            at::Tensor running_mean, at::Tensor running_var, const at::Tensor weight,
                            const at::Tensor bias, at::Tensor std, at::Tensor output);

void cudaSyncBNBackwardStep1(size_t n, size_t c, size_t h, size_t w, const at::Tensor input,
                             const at::Tensor mean, at::Tensor weight_diff, at::Tensor bias_diff,
                             const at::Tensor std, const at::Tensor grad_output);

void cudaSyncBNBackwardStep2(size_t n, size_t c, size_t h, size_t w, const at::Tensor input,
                            at::Tensor grad_input, const at::Tensor mean, const at::Tensor weight,
                            const at::Tensor weight_diff, const at::Tensor bias_diff, const at::Tensor std,
                            const at::Tensor grad_output);

void syncbn_forward_step1(const at::Tensor input, at::Tensor mean,
         size_t n, size_t c, size_t h, size_t w){
    cudaSyncBNForwardStep1(n, c, h, w, input, mean);
}

void syncbn_forward_step2(const at::Tensor input, at::Tensor mean, at::Tensor var,
         size_t n, size_t c, size_t h, size_t w){
    cudaSyncBNForwardStep2(n, c, h, w, input, mean, var);
}

void syncbn_forward_step3(const at::Tensor input, at::Tensor mean, at::Tensor var,
         const at::Tensor weight, const at::Tensor bias, at::Tensor running_mean,
         at::Tensor running_var, at::Tensor std, at::Tensor output,
         size_t n, size_t c, size_t h, size_t w, size_t group_size, const float eps,
         const float momentum){
    cudaSyncBNForwardStep3(n, c, h, w, group_size, input, eps, momentum, mean, var, running_mean,
                           running_var, weight, bias, std, output);
}

void syncbn_backward_step1(const at::Tensor input, const at::Tensor mean,
         const at::Tensor std, const at::Tensor grad_output, at::Tensor weight_diff,
         at::Tensor bias_diff, size_t n, size_t c, size_t h, size_t w){
    cudaSyncBNBackwardStep1(n, c, h, w, input, mean, weight_diff, bias_diff, std, grad_output);
}

void syncbn_backward_step2(const at::Tensor input, const at::Tensor mean,
         const at::Tensor weight, const at::Tensor weight_diff, const at::Tensor bias_diff,
         const at::Tensor std, const at::Tensor grad_output, at::Tensor grad_input, size_t n, size_t c, size_t h, size_t w){
    cudaSyncBNBackwardStep2(n, c, h, w, input, grad_input, mean, weight, weight_diff, bias_diff,
                            std, grad_output);
}
