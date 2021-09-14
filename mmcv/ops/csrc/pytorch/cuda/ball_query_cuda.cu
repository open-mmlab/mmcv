// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/ball_query_gpu.cu

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ball_query_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

void ball_query_kernel_launcher(int b, int n, int m, float min_radius,
                                float max_radius, int nsample,
                                const float *new_xyz, const float *xyz,
                                int *idx) {
  // new_xyz: (B, M, 3)
  // xyz: (B, N, 3)
  // output:
  //      idx: (B, M, nsample)

  cudaError_t err;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  dim3 blocks(DIVUP(m, THREADS_PER_BLOCK),
              b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  ball_query_cuda_kernel<<<blocks, threads, 0, stream>>>(
      b, n, m, min_radius, max_radius, nsample, new_xyz, xyz, idx);
  // cudaDeviceSynchronize();  // for using printf in kernel function
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
