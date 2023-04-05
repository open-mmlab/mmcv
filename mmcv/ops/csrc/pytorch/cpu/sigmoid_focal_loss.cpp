#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#include <cmath>
#include <iostream>
#include <cfloat>

template <typename T>
void sigmoid_focal_loss_forward_cpu_kernel(int N, const T *input, const int64_t *target,
                                           const T *weight, T *output,
                                           const float gamma,
                                           const float alpha, const int num_classes)
{
    for (int i = 0; i < N; i++)
    {
        int64_t t = target[i / num_classes];
        int c = i % num_classes;
        T p = (T)1. / ((T)1. + expf(-input[i]));
        T p_t = p * target[i] + (1 - p) * (1 - target[i]);
        if (t == c)
        {
            T term_p = pow(((T)1. - p), gamma) * log(std::max(p, (T)FLT_MIN));
            output[i] = -alpha * term_p;
        }
        else
        {
            T term_n = pow(p, gamma) * log(std::max((T)1. - p, (T)FLT_MIN));
            output[i] = -((T)1. - alpha) * term_n;
        }
        if (weight != NULL)
        {
            output[i] *= weight[t];
        }
    }
}

template <typename T>
void sigmoid_focal_loss_backward_cpu_kernel(const int N, const T *input, const int64_t *target,
                                            const T *weight, T *grad_input,
                                            const float gamma,
                                            const float alpha, const int num_classes)
{
    for (int i = 0; i < N; i++)
    {
        int64_t t = target[i / num_classes];
        int c = i % num_classes;
        T p = (T)1. / ((T)1. + expf(-input[i]));
        if (t == c)
        {
            T term_p = pow(((T)1. - p), gamma) *
                       ((T)1. - p - (gamma * p * log(std::max(p, (T)FLT_MIN))));
            grad_input[i] = -alpha * term_p;
        }
        else
        {
            T term_n = pow(p, gamma) *
                       (gamma * ((T)1. - p) * log(std::max((T)1. - p, (T)FLT_MIN)) - p);
            grad_input[i] = -((T)1. - alpha) * term_n;
        }
        if (weight != NULL)
        {
            grad_input[i] *= weight[t];
        }
    }
}

void TensorSigmoidFocalLossForwardCPUKernelLaucher(Tensor input, Tensor target,
                                                   Tensor weight, Tensor output,
                                                   const float gamma,
                                                   const float alpha)
{
    int output_size = output.numel();
    int num_classes = input.size(1);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "sigmoid_focal_loss_forward_cpu_kernel", [&]
        { sigmoid_focal_loss_forward_cpu_kernel(
              output_size, input.data_ptr<scalar_t>(),
              target.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>(), gamma, alpha, num_classes); });
}

void TensorSigmoidFocalLossBackwardCPUKernelLaucher(Tensor input, Tensor target,
                                                    Tensor weight, Tensor grad_input,
                                                    const float gamma,
                                                    const float alpha)
{
    int output_size = grad_input.numel();
    int num_classes = input.size(1);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "sigmoid_focal_loss_backward_cpu_kernel", [&]
        { sigmoid_focal_loss_backward_cpu_kernel<scalar_t>(
              output_size, input.data_ptr<scalar_t>(),
              target.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
              grad_input.data_ptr<scalar_t>(), gamma, alpha, num_classes); });
}

void sigmoid_focal_loss_forward_impl(Tensor input, Tensor target,
                                     Tensor weight, Tensor output,
                                     const float gamma,
                                     const float alpha);
void sigmoid_focal_loss_backward_impl(Tensor input, Tensor target,
                                      Tensor weight, Tensor grad_input,
                                      float gamma, float alpha);

REGISTER_DEVICE_IMPL(sigmoid_focal_loss_forward_impl, CPU, TensorSigmoidFocalLossForwardCPUKernelLaucher);
REGISTER_DEVICE_IMPL(sigmoid_focal_loss_backward_impl, CPU, TensorSigmoidFocalLossBackwardCPUKernelLaucher);
