// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/interpolate_gpu.cu

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pytorch_cuda_helper.hpp"
#include "pytorch_device_registry.hpp"
#include "three_nn_cuda_kernel.cuh"

void ThreeNNForwardCUDAKernelLauncher(int b, int n, int m, const Tensor unknown,
                                      const Tensor known, Tensor dist2,
                                      Tensor idx) {
  // unknown: (B, N, 3)
  // known: (B, M, 3)
  // output:
  //      dist2: (B, N, 3)
  //      idx: (B, N, 3)

  at::cuda::CUDAGuard device_guard(unknown.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), b);
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      unknown.scalar_type(), "three_nn_forward_cuda_kernel", [&] {
        three_nn_forward_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            b, n, m, unknown.data_ptr<scalar_t>(), known.data_ptr<scalar_t>(),
            dist2.data_ptr<scalar_t>(), idx.data_ptr<int>());
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void three_nn_forward_cuda(int b, int n, int m, const Tensor unknown,
                           const Tensor known, Tensor dist2, Tensor idx) {
  ThreeNNForwardCUDAKernelLauncher(b, n, m, unknown, known, dist2, idx);
};

void three_nn_forward_impl(int b, int n, int m, const Tensor unknown,
                           const Tensor known, Tensor dist2, Tensor idx);
REGISTER_DEVICE_IMPL(three_nn_forward_impl, CUDA, three_nn_forward_cuda);
