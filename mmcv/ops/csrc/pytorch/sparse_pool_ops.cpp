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
// #include <spconv/spconv/maxpool.h>
// #include <spconv/torch_utils.h>
// #include <torch/script.h>

#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
template <typename T>
torch::Tensor IndiceMaxpoolForwardCUDAKernelLauncher(torch::Tensor features,
                                                     torch::Tensor indicePairs,
                                                     torch::Tensor indiceNum,
                                                     int64_t numAct);

template <typename T>
torch::Tensor indice_maxpool_forward_cuda(torch::Tensor features,
                                          torch::Tensor indicePairs,
                                          torch::Tensor indiceNum,
                                          int64_t numAct) {
  return IndiceMaxpoolForwardCUDAKernelLauncher<T>(features, indicePairs,
                                                   indiceNum, numAct);
};

template <typename T>
torch::Tensor IndiceMaxpoolBackwardCUDAKernelLauncher(torch::Tensor features,
                                                      torch::Tensor outFeatures,
                                                      torch::Tensor outGrad,
                                                      torch::Tensor indicePairs,
                                                      torch::Tensor indiceNum);

template <typename T>
torch::Tensor indice_maxpool_backward_cuda(torch::Tensor features,
                                           torch::Tensor outFeatures,
                                           torch::Tensor outGrad,
                                           torch::Tensor indicePairs,
                                           torch::Tensor indiceNum) {
  return IndiceMaxpoolBackwardCUDAKernelLauncher<T>(
      features, outFeatures, outGrad, indicePairs, indiceNum);
};
#endif

template <typename T>
torch::Tensor indice_maxpool_forward(torch::Tensor features,
                                     torch::Tensor indicePairs,
                                     torch::Tensor indiceNum, int64_t numAct) {
  if (features.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(features);
    CHECK_CUDA_INPUT(indicePairs);
    CHECK_CUDA_INPUT(indiceNum);

    return indice_maxpool_forward_cuda<T>(features, indicePairs, indiceNum,
                                          numAct);
#else
    AT_ERROR("indice_maxpool is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("indice_maxpool is not implemented on CPU");
  }
}

template <typename T>
torch::Tensor indice_maxpool_backward(torch::Tensor features,
                                      torch::Tensor outFeatures,
                                      torch::Tensor outGrad,
                                      torch::Tensor indicePairs,
                                      torch::Tensor indiceNum) {
  if (features.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(features);
    CHECK_CUDA_INPUT(outFeatures);
    CHECK_CUDA_INPUT(outGrad);
    CHECK_CUDA_INPUT(indicePairs);
    CHECK_CUDA_INPUT(indiceNum);

    return indice_maxpool_backward_cuda<T>(features, outFeatures, outGrad,
                                           indicePairs, indiceNum);
#else
    AT_ERROR("indice_maxpool is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("indice_maxpool is not implemented on CPU");
  }
}
