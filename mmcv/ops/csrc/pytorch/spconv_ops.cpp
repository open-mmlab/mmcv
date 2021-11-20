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
template <unsigned NDim>
std::vector<torch::Tensor> GetIndicePairsForwardCUDAKernelLauncher(
    torch::Tensor indices, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose);

template <unsigned NDim>
std::vector<torch::Tensor> get_indice_pairs_forward_cuda(
    torch::Tensor indices, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose) {
  return GetIndicePairsForwardCUDAKernelLauncher<NDim>(
      indices, batchSize, outSpatialShape, spatialShape, kernelSize, stride,
      padding, dilation, outPadding, _subM, _transpose);
};

template <unsigned NDim>
std::vector<torch::Tensor> GetIndicePairsBackwardCUDAKernelLauncher(
    torch::Tensor indices, torch::Tensor gridOut, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose);

template <unsigned NDim>
std::vector<torch::Tensor> get_indice_pairs_backward_cuda(
    torch::Tensor indices, torch::Tensor gridOut, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose) {
  return GetIndicePairsBackwardCUDAKernelLauncher<NDim>(
      indices, gridOut, batchSize, outSpatialShape, spatialShape, kernelSize,
      stride, padding, dilation, outPadding, _subM, _transpose);
};

template <typename T>
torch::Tensor IndiceConvForwardCUDAKernelLauncher(
    torch::Tensor features, torch::Tensor filters, torch::Tensor indicePairs,
    torch::Tensor indiceNum, int64_t numActOut, int64_t _inverse,
    int64_t _subM);

template <typename T>
torch::Tensor indice_conv_forward_cuda(torch::Tensor features,
                                       torch::Tensor filters,
                                       torch::Tensor indicePairs,
                                       torch::Tensor indiceNum,
                                       int64_t numActOut, int64_t _inverse,
                                       int64_t _subM) {
  return IndiceConvForwardCUDAKernelLauncher<T>(
      features, filters, indicePairs, indiceNum, numActOut, _inverse, _subM);
};

template <typename T>
std::vector<torch::Tensor> IndiceConvBackwardCUDAKernelLauncher(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t _inverse,
    int64_t _subM);

template <typename T>
torch::Tensor indice_conv_backward_cuda(torch::Tensor features,
                                        torch::Tensor filters,
                                        torch::Tensor outGrad,
                                        torch::Tensor indicePairs,
                                        torch::Tensor indiceNum,
                                        int64_t _inverse, int64_t _subM) {
  return IndiceConvBackwardCUDAKernelLauncher<T>(
      features, filters, outGrad, indicePairs, indiceNum, _inverse, _subM);
};
#endif

template <unsigned NDim>
std::vector<torch::Tensor> get_indice_pairs_forward(
    torch::Tensor indices, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose) {
  if (indices.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(indices);

    return get_indice_pairs_forward_cuda<NDim>(
        indices, batchSize, outSpatialShape, spatialShape, kernelSize, stride,
        padding, dilation, outPadding, _subM, _transpose);
#else
    AT_ERROR("get_indice_pairs is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("get_indice_pairs is not implemented on CPU");
  }
}

template <unsigned NDim>
std::vector<torch::Tensor> get_indice_pairs_backward(
    torch::Tensor indices, torch::Tensor gridOut, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose) {
  if (indices.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(indices);
    CHECK_CUDA_INPUT(gridOut);

    return get_indice_pairs_backward_cuda<NDim>(
        indices, gridOut, batchSize, outSpatialShape, spatialShape, kernelSize,
        stride, padding, dilation, outPadding, _subM, _transpose);
#else
    AT_ERROR("get_indice_pairs is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("get_indice_pairs is not implemented on CPU");
  }
}

template <typename T>
torch::Tensor indice_conv_forward(torch::Tensor features, torch::Tensor filters,
                                  torch::Tensor indicePairs,
                                  torch::Tensor indiceNum, int64_t numActOut,
                                  int64_t _inverse, int64_t _subM) {
  if (features.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(features);
    CHECK_CUDA_INPUT(filters);
    CHECK_CUDA_INPUT(indicePairs);
    CHECK_CUDA_INPUT(indiceNum);

    return indice_conv_forward_cuda<T>(features, filters, indicePairs,
                                       indiceNum, numActOut, _inverse, _subM);
#else
    AT_ERROR("indice_conv is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("indice_conv is not implemented on CPU");
  }
}

template <typename T>
std::vector<torch::Tensor> indice_conv_backward(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t _inverse,
    int64_t _subM) {
  if (features.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(features);
    CHECK_CUDA_INPUT(filters);
    CHECK_CUDA_INPUT(outGrad);
    CHECK_CUDA_INPUT(indicePairs);
    CHECK_CUDA_INPUT(indiceNum);

    return indice_conv_backward_cuda<T>(features, filters, outGrad, indicePairs,
                                        indiceNum, _inverse, _subM);
#else
    AT_ERROR("indice_conv is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("indice_conv is not implemented on CPU");
  }
}
