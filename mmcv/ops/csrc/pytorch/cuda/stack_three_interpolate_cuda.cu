// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/interpolate_gpu.cu

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pytorch_cuda_helper.hpp"
#include "stack_three_interpolate_cuda_kernel.cuh"

void StackThreeInterpolateForwardCUDAKernelLauncher(
                                               const Tensor points,
                                               const Tensor idx,
                                               const Tensor weight,
                                               Tensor out) {
  // points: (B, C, M)
  // idx: (B, N, 3)
  // weight: (B, N, 3)
  // output:
  //      out: (B, C, N)

  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // blockIdx.x(col), blockIdx.y(row)
      int N = out.size(0);
int channels = points.size(1);
    dim3 blocks(DIVUP(N, THREADS_PER_BLOCK), channels);
    dim3 threads(THREADS_PER_BLOCK);


stack_three_interpolate_forward_cuda_kernel<<<blocks, threads>>>(N, channels,points.data_ptr<float>(), idx.data_ptr<int>(),
                weight.data_ptr<float>(), out.data_ptr<float>());
//   AT_DISPATCH_FLOATING_TYPES_AND_HALF(
//       points.scalar_type(), "stack_three_interpolate_forward_cuda_kernel", [&] {
//         three_interpolate_forward_cuda_kernel<scalar_t>
//             <<<blocks, threads, 0, stream>>>(
//             N, channels,
//                 points.data_ptr<scalar_t>(), idx.data_ptr<int>(),
//                 weight.data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
//       });

  AT_CUDA_CHECK(cudaGetLastError());
}

void StackThreeInterpolateBackwardCUDAKernelLauncher(
                                                const Tensor grad_out,
                                                const Tensor idx,
                                                const Tensor weight,
                                                Tensor grad_points) {
  // grad_out: (B, C, N)
  // weight: (B, N, 3)
  // output:
  //      grad_points: (B, C, M)

  at::cuda::CUDAGuard device_guard(grad_out.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // blockIdx.x(col), blockIdx.y(row)
    int N = grad_out.size(0);
    int channels = grad_out.size(1);
        dim3 blocks(DIVUP(N, THREADS_PER_BLOCK), channels);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    stack_three_interpolate_backward_cuda_kernel<<<blocks, threads>>>(
                    N, channels, grad_out.data_ptr<float>(), idx.data_ptr<int>(),
                weight.data_ptr<float>(), grad_points.data_ptr<float>()
    );
//   AT_DISPATCH_FLOATING_TYPES_AND_HALF(
//       grad_out.scalar_type(), "stack_three_interpolate_backward_cuda_kernel", [&] {
//         three_interpolate_backward_cuda_kernel<scalar_t>
//             <<<blocks, threads, 0, stream>>>(
//                 N, channels, grad_out.data_ptr<scalar_t>(), idx.data_ptr<int>(),
//                 weight.data_ptr<scalar_t>(), grad_points.data_ptr<scalar_t>());
//       });

  AT_CUDA_CHECK(cudaGetLastError());
}
