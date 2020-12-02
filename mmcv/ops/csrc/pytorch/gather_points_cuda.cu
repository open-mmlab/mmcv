#include "gather_points_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

#define TOTAL_THREADS 1024

void GatherPointsCUDAKernelLauncher(int b, int c, int n, int npoints,
                                    const Tensor points, const Tensor idx,
                                    Tensor out) {
  // points: (B, C, N)
  // idx: (B, npoints)
  // output:
  //      out: (B, C, npoints)

  cudaError_t err;

  dim3 blocks(GET_BLOCKS(npoints), c, b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  gather_points_cuda_kernel<<<blocks, threads, 0, stream>>>(
      b, c, n, npoints, points.data_ptr<float>(), idx.data_ptr<int>(),
      out.data_ptr<float>());

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

void GatherPointsBackwardCUDAKernelLauncher(int b, int c, int n, int npoints,
                                            const Tensor grad_out,
                                            const Tensor idx,
                                            Tensor grad_points) {
  // grad_out: (B, C, npoints)
  // idx: (B, npoints)
  // output:
  //      grad_points: (B, C, N)

  cudaError_t err;

  dim3 blocks(GET_BLOCKS(npoints), c, b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  gather_points_backward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      b, c, n, npoints, grad_out.data_ptr<float>(), idx.data_ptr<int>(),
      grad_points.data_ptr<float>());

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
