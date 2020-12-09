// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/interpolate_gpu.cu

#include "pytorch_cuda_helper.hpp"
#include "three_interpolate_cuda_kernel.cuh"

void ThreeInterpolateCUDAKernelLauncher(int b, int c, int m, int n,
                                        const Tensor points, const Tensor idx,
                                        const Tensor weight, Tensor out) {
  // points: (B, C, M)
  // idx: (B, N, 3)
  // weight: (B, N, 3)
  // output:
  //      out: (B, C, N)

  cudaError_t err;
  dim3 blocks(GET_BLOCKS(n), c, b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  three_interpolate_cuda_kernel<<<blocks, threads, 0, stream>>>(
      b, c, m, n, points.data_ptr<float>(), idx.data_ptr<int>(),
      weight.data_ptr<float>(), out.data_ptr<float>());

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

void ThreeInterpolateBackwardCUDAKernelLauncher(int b, int c, int n, int m,
                                                const Tensor grad_out,
                                                const Tensor idx,
                                                const Tensor weight,
                                                Tensor grad_points) {
  // grad_out: (B, C, N)
  // weight: (B, N, 3)
  // output:
  //      grad_points: (B, C, M)

  cudaError_t err;
  dim3 blocks(GET_BLOCKS(n), c, b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  three_interpolate_backward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      b, c, n, m, grad_out.data_ptr<float>(), idx.data_ptr<int>(),
      weight.data_ptr<float>(), grad_points.data_ptr<float>());

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

void ThreeNNCUDAKernelLauncher(int b, int n, int m, const Tensor unknown,
                               const Tensor known, Tensor dist2, Tensor idx) {
  // unknown: (B, N, 3)
  // known: (B, M, 3)
  // output:
  //      dist2: (B, N, 3)
  //      idx: (B, N, 3)

  cudaError_t err;
  dim3 blocks(GET_BLOCKS(n), b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  three_nn_cuda_kernel<<<blocks, threads, 0, stream>>>(
      b, n, m, unknown.data_ptr<float>(), known.data_ptr<float>(),
      dist2.data_ptr<float>(), idx.data_ptr<int>());

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
