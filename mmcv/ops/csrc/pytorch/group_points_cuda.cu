// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/group_points_gpu.cu
#include "group_points_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

void GroupPointsBackwardCUDAKernelLauncher(int b, int c, int n, int npoints,
                                           int nsample, const Tensor grad_out,
                                           const Tensor idx,
                                           Tensor grad_points) {
  // grad_out: (B, C, npoints, nsample)
  // idx: (B, npoints, nsample)
  // output:
  //      grad_points: (B, C, N)
  cudaError_t err;
  dim3 blocks(GET_BLOCKS(npoints * nsample), c,
              b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  group_points_backward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      b, c, n, npoints, nsample, grad_out.data_ptr<float>(),
      idx.data_ptr<int>(), grad_points.data_ptr<float>());

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

void GroupPointsCUDAKernelLauncher(int b, int c, int n, int npoints,
                                   int nsample, const Tensor points,
                                   const Tensor idx, Tensor out) {
  // points: (B, C, N)
  // idx: (B, npoints, nsample)
  // output:
  //      out: (B, C, npoints, nsample)
  cudaError_t err;
  dim3 blocks(GET_BLOCKS(npoints * nsample), c,
              b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  group_points_cuda_kernel<<<blocks, threads, 0, stream>>>(
      b, c, n, npoints, nsample, points.data_ptr<float>(), idx.data_ptr<int>(),
      out.data_ptr<float>());
  // cudaDeviceSynchronize();  // for using printf in kernel function
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
