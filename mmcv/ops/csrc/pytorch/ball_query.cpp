// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/ball_query.cpp

#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <vector>

#include "pytorch_cpp_helper.hpp"

extern THCState *state;

int ball_query_wrapper(int b, int n, int m, float min_radius, float max_radius, int nsample,
                       at::Tensor new_xyz_tensor, at::Tensor xyz_tensor,
                       at::Tensor idx_tensor);

void ball_query_kernel_launcher(int b, int n, int m, float min_radius, float max_radius,
                                int nsample, const float *xyz, const float *new_xyz,
                                int *idx, cudaStream_t stream);

int ball_query_wrapper(int b, int n, int m, float min_radius, float max_radius, int nsample,
                       at::Tensor new_xyz_tensor, at::Tensor xyz_tensor,
                       at::Tensor idx_tensor)
{
    CHECK_CUDA_INPUT(new_xyz_tensor);
    CHECK_CUDA_INPUT(xyz_tensor);
    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    ball_query_kernel_launcher(b, n, m, min_radius, max_radius,
                               nsample, new_xyz, xyz, idx, stream);
    return 1;
}