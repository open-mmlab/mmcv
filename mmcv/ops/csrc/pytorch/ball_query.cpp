// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/ball_query.cpp
#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void BallQueryCUDAKernelLauncher(int b, int n, int m, float min_radius,
                                 float max_radius, int nsample,
                                 const Tensor new_xyz, const Tensor xyz,
                                 Tensor idx);

void ball_query_cuda(int b, int n, int m, float min_radius, float max_radius,
                     int nsample, const Tensor new_xyz, const Tensor xyz,
                     Tensor idx) {
  BallQueryCUDAKernelLauncher(b, n, m, min_radius, max_radius, nsample, new_xyz,
                              xyz, idx);
}
#endif

int ball_query(int b, int n, int m, float min_radius, float max_radius,
               int nsample, const Tensor new_xyz, const Tensor xyz,
               Tensor idx) {
  if (new_xyz.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(new_xyz);
    CHECK_CUDA_INPUT(xyz);

    ball_query_cuda(b, n, m, min_radius, max_radius, nsample, new_xyz, xyz,
                    idx);
    return 1;
#else
    AT_ERROR("ball_query is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("ball_query is not implemented on CPU");
  }
}
