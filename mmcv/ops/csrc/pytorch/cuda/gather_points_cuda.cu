#include <stdio.h>
#include <stdlib.h>

#include "gather_points_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

#define TOTAL_THREADS 1024
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

void gather_points_cuda_forward(int b, int c, int n, int npoints,
                                const float *points, const int *idx,
                                float *out) {
  // points: (B, C, N)
  // idx: (B, npoints)
  // output:
  //      out: (B, C, npoints)

  cudaError_t err;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  dim3 blocks(DIVUP(npoints, THREADS_PER_BLOCK), c,
              b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  gather_points_kernel<<<blocks, threads, 0, stream>>>(b, c, n, npoints, points,
                                                       idx, out);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

void gather_points_cuda_backward(int b, int c, int n, int npoints,
                                 const float *grad_out, const int *idx,
                                 float *grad_points) {
  // grad_out: (B, C, npoints)
  // idx: (B, npoints)
  // output:
  //      grad_points: (B, C, N)

  cudaError_t err;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  dim3 blocks(DIVUP(npoints, THREADS_PER_BLOCK), c,
              b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  gather_points_grad_kernel<<<blocks, threads, 0, stream>>>(
      b, c, n, npoints, grad_out, idx, grad_points);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
