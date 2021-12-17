// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/csuhan/s2anet/blob/master/mmdet/ops/orn/src/cpu/ActiveRotatingFilter_cpu.cpp
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

template <typename T>
void ARF_forward_cpu_kernel(const T* weightData, const int* indicesData,
                            const int nOutputPlane, const int nInputPlane,
                            const int nOrientation, const int kH, const int kW,
                            const int nRotation, T* outputData) {
  const int nEntry = nOrientation * kH * kW;
  int i, j, l;
  int k;

#pragma omp parallel for private(i, j, l, k)
  for (i = 0; i < nOutputPlane; i++) {
    for (j = 0; j < nInputPlane; j++) {
      for (l = 0; l < nEntry; l++) {
        int weightIndex = i * nInputPlane * nEntry + j * nEntry + l;
        T val = *(weightData + weightIndex);
        // T val = *(weightData++);
        for (k = 0; k < nRotation; k++) {
          int index = (int)(*(indicesData + l * nRotation + k)) - 1;
          T* target = outputData + i * (nRotation * nInputPlane * nEntry) +
                      k * (nInputPlane * nEntry) + j * (nEntry) + index;
          *target = val;
        }
      }
    }
  }
}

template <typename T>
void ARF_backward_cpu_kernel(const int* indicesData, const T* gradOutputData,
                             const int nOutputPlane, const int nInputPlane,
                             const int nOrientation, const int kH, const int kW,
                             const int nRotation, T* gradInputData) {
  const int nEntry = nOrientation * kH * kW;
  int i, j, l;
  int k;

#pragma omp parallel for private(i, j, l, k)
  for (i = 0; i < nOutputPlane; i++) {
    for (j = 0; j < nInputPlane; j++) {
      for (l = 0; l < nEntry; l++) {
        int gradInputIndex = i * nInputPlane * nEntry + j * nEntry + l;
        T* val = gradInputData + gradInputIndex;
        // T *val = gradInputData++;
        *val = 0;
        for (k = 0; k < nRotation; k++) {
          int index = (int)(*(indicesData + l * nRotation + k)) - 1;
          const T* target = gradOutputData +
                            i * (nRotation * nInputPlane * nEntry) +
                            k * (nInputPlane * nEntry) + j * (nEntry) + index;
          *val = *val + *target;
        }
      }
    }
  }
}

void active_rotated_filter_forward_cpu(const Tensor input, const Tensor indices,
                                       Tensor output) {
  const int nOutputPlane = input.size(0);
  const int nInputPlane = input.size(1);
  const int nOrientation = input.size(2);
  const int kH = input.size(3);
  const int kW = input.size(4);
  const int nRotation = indices.size(3);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "ARF_forward", [&] {
    ARF_forward_cpu_kernel<scalar_t>(input.data_ptr<scalar_t>(),
                                     indices.data_ptr<int>(), nOutputPlane,
                                     nInputPlane, nOrientation, kH, kW,
                                     nRotation, output.data_ptr<scalar_t>());
  });
}

void active_rotated_filter_backward_cpu(const Tensor grad_out,
                                        const Tensor indices, Tensor grad_in) {
  const int nOrientation = indices.size(0);
  const int kH = indices.size(1);
  const int kW = indices.size(2);
  const int nRotation = indices.size(3);
  const int nOutputPlane = grad_out.size(0) / nRotation;
  const int nInputPlane = grad_out.size(1) / nOrientation;

  AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "ARF_backward", [&] {
    ARF_backward_cpu_kernel<scalar_t>(
        indices.data_ptr<int>(), grad_out.data_ptr<scalar_t>(), nOutputPlane,
        nInputPlane, nOrientation, kH, kW, nRotation,
        grad_in.data_ptr<scalar_t>());
  });
}

void active_rotated_filter_forward_impl(const Tensor input,
                                        const Tensor indices, Tensor output);

void active_rotated_filter_backward_impl(const Tensor grad_out,
                                         const Tensor indices, Tensor grad_in);

REGISTER_DEVICE_IMPL(active_rotated_filter_forward_impl, CPU,
                     active_rotated_filter_forward_cpu);
REGISTER_DEVICE_IMPL(active_rotated_filter_backward_impl, CPU,
                     active_rotated_filter_backward_cpu);
