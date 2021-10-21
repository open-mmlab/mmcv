// Copyright (c) OpenMMLab. All rights reserved.
#include "pytorch_cpp_helper.hpp"

typedef enum { SUM = 0, MEAN = 1, MAX = 2 } reduce_t;

#ifdef MMCV_WITH_CUDA
std::vector<torch::Tensor> DynamicPointToVoxelForwardCUDAKernelLauncher(
    const torch::Tensor &feats, const torch::Tensor &coors,
    const reduce_t reduce_type);

std::vector<torch::Tensor> dynamic_point_to_voxel_forward_cuda(
    const torch::Tensor &feats, const torch::Tensor &coors,
    const reduce_t reduce_type) {
  return DynamicPointToVoxelForwardCUDAKernelLauncher(feats, coors,
                                                      reduce_type);
};

void DynamicPointToVoxelBackwardCUDAKernelLauncher(
    torch::Tensor &grad_feats, const torch::Tensor &grad_reduced_feats,
    const torch::Tensor &feats, const torch::Tensor &reduced_feats,
    const torch::Tensor &coors_idx, const torch::Tensor &reduce_count,
    const reduce_t reduce_type);

void dynamic_point_to_voxel_backward_cuda(
    torch::Tensor &grad_feats, const torch::Tensor &grad_reduced_feats,
    const torch::Tensor &feats, const torch::Tensor &reduced_feats,
    const torch::Tensor &coors_idx, const torch::Tensor &reduce_count,
    const reduce_t reduce_type) {
  DynamicPointToVoxelBackwardCUDAKernelLauncher(grad_feats, grad_reduced_feats,
                                                feats, reduced_feats, coors_idx,
                                                reduce_count, reduce_type);
};
#endif

std::vector<at::Tensor> dynamic_point_to_voxel_forward_cpu(
    const at::Tensor &points, const at::Tensor &voxel_mapping,
    const std::vector<float> voxel_size, const std::vector<float> coors_range);

inline reduce_t convert_reduce_type(const std::string &reduce_type) {
  if (reduce_type == "max")
    return reduce_t::MAX;
  else if (reduce_type == "sum")
    return reduce_t::SUM;
  else if (reduce_type == "mean")
    return reduce_t::MEAN;
  else
    TORCH_CHECK(false, "do not support reduce type " + reduce_type)
  return reduce_t::SUM;
}

std::vector<torch::Tensor> dynamic_point_to_voxel_forward(
    const torch::Tensor &feats, const torch::Tensor &coors,
    const std::string &reduce_type) {
  if (feats.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(feats);
    CHECK_CUDA_INPUT(coors);
    return dynamic_point_to_voxel_forward_cuda(
        feats, coors, convert_reduce_type(reduce_type));
#else
    AT_ERROR("dynamic_point_to_voxel is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("dynamic_point_to_voxel is not implemented on CPU");
    return std::vector<torch::Tensor>();
  }
}

void dynamic_point_to_voxel_backward(torch::Tensor &grad_feats,
                                     const torch::Tensor &grad_reduced_feats,
                                     const torch::Tensor &feats,
                                     const torch::Tensor &reduced_feats,
                                     const torch::Tensor &coors_idx,
                                     const torch::Tensor &reduce_count,
                                     const std::string &reduce_type) {
  if (grad_feats.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(grad_feats);
    CHECK_CUDA_INPUT(grad_reduced_feats);
    CHECK_CUDA_INPUT(feats);
    CHECK_CUDA_INPUT(reduced_feats);
    CHECK_CUDA_INPUT(coors_idx);
    CHECK_CUDA_INPUT(reduce_count);
    dynamic_point_to_voxel_backward_cuda(grad_feats, grad_reduced_feats, feats,
                                         reduced_feats, coors_idx, reduce_count,
                                         convert_reduce_type(reduce_type));
#else
    AT_ERROR("dynamic_point_to_voxel is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("dynamic_point_to_voxel is not implemented on CPU");
  }
}
