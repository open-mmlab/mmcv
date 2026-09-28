// Copyright (c) OpenMMLab. All rights reserved
#ifndef SIGMOID_FOCAL_LOSS_CUDA_KERNEL_CUH
#define SIGMOID_FOCAL_LOSS_CUDA_KERNEL_CUH

#include <type_traits>

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename T>
__global__ void sigmoid_focal_loss_forward_cuda_kernel(
    const int nthreads, const T* input, const int64_t* target, const T* weight,
    T* output, const float gamma, const float alpha, const int num_classes) {
  // Evaluate half inputs in float, while preserving double precision.
  using acc_t = typename std::conditional<std::is_same<T, double>::value,
                                          double, float>::type;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / num_classes;
    int c = index % num_classes;

    int64_t t = target[n];
    const acc_t x = input[index];
    const acc_t gamma_ = gamma;
    const acc_t alpha_ = alpha;
    const acc_t z = exp(-abs(x));
    const acc_t p = x >= 0 ? acc_t(1) / (acc_t(1) + z) : z / (acc_t(1) + z);
    const acc_t q = x >= 0 ? z / (acc_t(1) + z) : acc_t(1) / (acc_t(1) + z);
    // Compute log probabilities from logits, without log(0) or a clamp that
    // underflows in half precision. This also preserves large finite losses.
    const acc_t log_p = (x >= 0 ? acc_t(0) : x) - log1p(z);
    const acc_t log_q = (x >= 0 ? -x : acc_t(0)) - log1p(z);
    acc_t loss = t == c ? -alpha_ * pow(q, gamma_) * log_p
                        : -(acc_t(1) - alpha_) * pow(p, gamma_) * log_q;
    if (weight != NULL) {
      loss *= acc_t(weight[t]);
    }
    output[index] = T(loss);
  }
}

template <typename T>
__global__ void sigmoid_focal_loss_backward_cuda_kernel(
    const int nthreads, const T* input, const int64_t* target, const T* weight,
    T* grad_input, const float gamma, const float alpha,
    const int num_classes) {
  using acc_t = typename std::conditional<std::is_same<T, double>::value,
                                          double, float>::type;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / num_classes;
    int c = index % num_classes;

    int64_t t = target[n];
    const acc_t x = input[index];
    const acc_t gamma_ = gamma;
    const acc_t alpha_ = alpha;
    const acc_t z = exp(-abs(x));
    const acc_t p = x >= 0 ? acc_t(1) / (acc_t(1) + z) : z / (acc_t(1) + z);
    const acc_t q = x >= 0 ? z / (acc_t(1) + z) : acc_t(1) / (acc_t(1) + z);
    const acc_t log_p = (x >= 0 ? acc_t(0) : x) - log1p(z);
    const acc_t log_q = (x >= 0 ? -x : acc_t(0)) - log1p(z);
    acc_t grad = t == c ? -alpha_ * pow(q, gamma_) * (q - gamma_ * p * log_p)
                        : -(acc_t(1) - alpha_) * pow(p, gamma_) *
                              (gamma_ * q * log_q - p);
    if (weight != NULL) {
      grad *= acc_t(weight[t]);
    }
    grad_input[index] = T(grad);
  }
}

#endif  // SIGMOID_FOCAL_LOSS_CUDA_KERNEL_CUH
