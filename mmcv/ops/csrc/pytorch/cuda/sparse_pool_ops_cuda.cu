#include <cuda_runtime_api.h>
#include <spconv/spconv/maxpool.h>
#include <spconv/torch_utils.h>
#include <torch/script.h>

#include "pytorch_cuda_helper.hpp"

template <typename T>
torch::Tensor IndiceMaxpoolForwardCUDAKernelLauncher(torch::Tensor features,
                                                     torch::Tensor indicePairs,
                                                     torch::Tensor indiceNum,
                                                     int64_t numAct) {
  auto device = features.device().type();
  auto kernelVolume = indicePairs.size(0);
  auto numInPlanes = features.size(1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  torch::Tensor output = torch::zeros({numAct, numInPlanes}, options);
  double totalTime = 0;
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    if (nHot <= 0) {
      continue;
    }
    if (device == torch::kCPU) {
      functor::SparseMaxPoolForwardFunctor<tv::CPU, T, int> forwardFtor;
      forwardFtor(tv::CPU(), tv::torch2tv<T>(output),
                  tv::torch2tv<const T>(features),
                  tv::torch2tv<const int>(indicePairs).subview(i), nHot);
    } else {
      functor::SparseMaxPoolForwardFunctor<tv::GPU, T, int> forwardFtor;
      forwardFtor(tv::TorchGPU(), tv::torch2tv<T>(output),
                  tv::torch2tv<const T>(features),
                  tv::torch2tv<const int>(indicePairs).subview(i), nHot);
      TV_CHECK_CUDA_ERR();
    }
  }
  return output;
}

template <typename T>
torch::Tensor IndiceMaxpoolBackwardCUDAKernelLauncher(torch::Tensor features,
                                                      torch::Tensor outFeatures,
                                                      torch::Tensor outGrad,
                                                      torch::Tensor indicePairs,
                                                      torch::Tensor indiceNum) {
  auto device = features.device().type();
  auto numInPlanes = features.size(1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  torch::Tensor inputGrad = torch::zeros(features.sizes(), options);
  auto kernelVolume = indicePairs.size(0);
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    if (nHot <= 0) {
      continue;
    }
    if (device == torch::kCPU) {
      functor::SparseMaxPoolBackwardFunctor<tv::CPU, T, int> backwardFtor;
      backwardFtor(tv::CPU(), tv::torch2tv<const T>(outFeatures),
                   tv::torch2tv<const T>(features),
                   tv::torch2tv<const T>(outGrad), tv::torch2tv<T>(inputGrad),
                   tv::torch2tv<const int>(indicePairs).subview(i), nHot);
    } else {
      functor::SparseMaxPoolBackwardFunctor<tv::GPU, T, int> backwardFtor;
      backwardFtor(tv::TorchGPU(), tv::torch2tv<const T>(outFeatures),
                   tv::torch2tv<const T>(features),
                   tv::torch2tv<const T>(outGrad), tv::torch2tv<T>(inputGrad),
                   tv::torch2tv<const int>(indicePairs).subview(i), nHot);
      TV_CHECK_CUDA_ERR();
    }
  }
  return inputGrad;
}

template torch::Tensor IndiceMaxpoolForwardCUDAKernelLauncher<float>(
    torch::Tensor features, torch::Tensor indicePairs, torch::Tensor indiceNum,
    int64_t numAct);

template torch::Tensor IndiceMaxpoolForwardCUDAKernelLauncher<at::Half>(
    torch::Tensor features, torch::Tensor indicePairs, torch::Tensor indiceNum,
    int64_t numAct);

template torch::Tensor IndiceMaxpoolBackwardCUDAKernelLauncher<float>(
    torch::Tensor features, torch::Tensor outFeatures, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum);

template torch::Tensor IndiceMaxpoolBackwardCUDAKernelLauncher<at::Half>(
    torch::Tensor features, torch::Tensor outFeatures, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum);
