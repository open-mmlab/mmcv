// Copyright (c) OpenMMLab. All rights reserved
#ifndef SOFTMAX_FOCAL_LOSS_CUDA_KERNEL_CUH
#define SOFTMAX_FOCAL_LOSS_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename T>
__global__ void softmax_focal_loss_forward_cuda_kernel(
    const int nthreads, const T* __restrict__ log_softmax_prob,
    const int64_t* __restrict__ target, const T* __restrict__ weight,
    T* __restrict__ output,
    const T gamma, const T alpha, const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int n = index / num_classes;
    const int c = index % num_classes;

    // focal loss
    // FL(p) = - alpha * (1-p)^gamma * log(p) if curr_class == label
    //
    // note that log_softmax_prob is calculated in Python part
    // by using PyTorch API F.log_softmax()
    const int64_t label = target[n];
    if (c == label) {
      const T w = (weight != NULL) ? weight[label] : T(1);
      const T alpha_fac = ((label == 0) * (1 - alpha) + (label >= 1) * alpha) * w;

      const T log_pred = log_softmax_prob[index];
      const T pred = exp(log_pred);

      output[index] = -alpha_fac * pow(1 - pred, gamma) * log_pred;
    } else {
      output[index] = 0;
    }
  }
}

template <typename T>
__global__ void softmax_focal_loss_backward_cuda_kernel(
    const int nthreads, const T* __restrict__ log_softmax_prob,
    const int64_t* __restrict__ target, const T* __restrict__ weight,
    T* __restrict__ sum_buff_along_class, T* __restrict__ grad_input,
    const T gamma, const T alpha, const int num_classes) {
    // forward node:  x ----> p ----> FL
    //         func:     SM      FL
    //
    // backward node: x <---- p <---- FL
    //         index: j       i       FL
    //
    // For simplicity, the alpha of FL is ignored here
    // dFL/dp = - [((1-p)^gamma) / p
    //             - gamma * (1-p)^(gamma-1) * log(p)]
    // dp_i/dx_j = dSM/dx_j
    //           = p_i * (1-p_j) i==j;
    //             p_i * (0-p_j) i!=j;
    //           = p_i * (delta - p_j) where delta is Kronecker delta
    //
    // Replacing the p of dFL/dp with p_i, then
    // dFL/dx_j = dFL/dp_i * dp_i/dx_j
    //          = - (delta - p_j) * [ (1-p_i)^gamma
    //              - gamma * (1-p_i)^(gamma-1) * log(p) * p_i]
    //          =   (delta - p_j) * [- (1-p_i)^gamma +
    //                gamma * (1-p_i)^(gamma-1) * log(p) * p_i]
    //
    // Let B_i denote [- (1-p_i)^gamma +
    //   gamma * (1-p_i)^(gamma-1) * log(p) * p_i],
    // and indices {i} is summed for all classes at index j
    // since x_j received all the gradients from {p_i}.
    // Then, dFL/dx_j = sum_i{ (delta - p_j) * B_i }
    //                = sum_i{  delta*B_i - p_j*B_i }
    //                = B_j - (p_j * sum_i{B_i})

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // B_i
    const int n = index / num_classes;
    const int c = index % num_classes;

    const int64_t label = target[n];
    if (c == label) {
      const T w = (weight != NULL) ? weight[label] : T(1);
      const T alpha_fac = ((label == 0) * (1 - alpha) + (label >= 1) * alpha) * w;

      const T log_pred = log_softmax_prob[index];
      const T pred = exp(log_pred);
      const T one_minus_pred = 1 - pred;

      const T buff = alpha_fac * (
        -pow(one_minus_pred, gamma) +
        gamma * pow(one_minus_pred, gamma - 1) * log_pred * pred
      );
      grad_input[index] = buff;
      sum_buff_along_class[n] += buff;
    } else {
      grad_input[index] = 0;
    }
  }

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // dFL/dx_j
    const int n = index / num_classes;

    const T pred = exp(log_softmax_prob[index]);
    grad_input[index] -= pred * sum_buff_along_class[n];
  }
}

#endif  // SOFTMAX_FOCAL_LOSS_CUDA_KERNEL_CUH
