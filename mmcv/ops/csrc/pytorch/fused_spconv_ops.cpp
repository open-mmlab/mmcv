// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// #include <cuda_runtime_api.h>
// #include <spconv/spconv/indice.h>
// #include <spconv/spconv/reordering.h>
// #include <spconv/torch_utils.h>
// #include <torch/script.h>

#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
template <typename T>
torch::Tensor FusedIndiceConvBatchnormCUDAKernelLauncher(
    torch::Tensor features, torch::Tensor filters, torch::Tensor bias,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numActOut,
    int64_t _inverse, int64_t _subM);

template <typename T>
torch::Tensor fused_indice_conv_batchnorm_forward_cuda(
    torch::Tensor features, torch::Tensor filters, torch::Tensor bias,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numActOut,
    int64_t _inverse, int64_t _subM) {
  return FusedIndiceConvBatchnormCUDAKernelLauncher<T>(
      features, filters, bias, indicePairs, indiceNum, numActOut, _inverse,
      _subM);
};
#endif

template <typename T>
torch::Tensor fused_indice_conv_batchnorm_forward(
    torch::Tensor features, torch::Tensor filters, torch::Tensor bias,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numActOut,
    int64_t _inverse, int64_t _subM) {
  if (features.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(features);
    CHECK_CUDA_INPUT(filters);
    CHECK_CUDA_INPUT(bias);
    CHECK_CUDA_INPUT(indicePairs);
    CHECK_CUDA_INPUT(indiceNum);

    return fused_indice_conv_batchnorm_forward_cuda<T>(
        features, filters, bias, indicePairs, indiceNum, numActOut, _inverse,
        _subM);
#else
    AT_ERROR("fused_indice_conv_batchnorm is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("fused_indice_conv_batchnorm is not implemented on CPU");
  }
}
