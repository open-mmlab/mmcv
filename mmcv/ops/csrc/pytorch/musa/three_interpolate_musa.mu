// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/interpolate_gpu.cu

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pytorch_musa_helper.hpp"
#include "three_interpolate_musa_kernel.muh"

void ThreeInterpolateForwardMUSAKernelLauncher(int b, int c, int m, int n,
                                               const Tensor points,
                                               const Tensor idx,
                                               const Tensor weight,
                                               Tensor out) {
  // points: (B, C, M)
  // idx: (B, N, 3)
  // weight: (B, N, 3)
  // output:
  //      out: (B, C, N)

  at::musa::MUSAGuard device_guard(points.device());
  musaStream_t stream = at::musa::getCurrentMUSAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(GET_BLOCKS(n, THREADS_PER_BLOCK), c, b);
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "three_interpolate_forward_musa_kernel", [&] {
        three_interpolate_forward_musa_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                b, c, m, n, points.data_ptr<scalar_t>(), idx.data_ptr<int>(),
                weight.data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
      });

  AT_MUSA_CHECK(musaGetLastError());
}

void ThreeInterpolateBackwardMUSAKernelLauncher(int b, int c, int n, int m,
                                                const Tensor grad_out,
                                                const Tensor idx,
                                                const Tensor weight,
                                                Tensor grad_points) {
  // grad_out: (B, C, N)
  // weight: (B, N, 3)
  // output:
  //      grad_points: (B, C, M)

  at::musa::MUSAGuard device_guard(grad_out.device());
  musaStream_t stream = at::musa::getCurrentMUSAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(GET_BLOCKS(n, THREADS_PER_BLOCK), c, b);
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES(
      grad_out.scalar_type(), "three_interpolate_backward_musa_kernel", [&] {
        three_interpolate_backward_musa_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                b, c, n, m, grad_out.data_ptr<scalar_t>(), idx.data_ptr<int>(),
                weight.data_ptr<scalar_t>(), grad_points.data_ptr<scalar_t>());
      });

  AT_MUSA_CHECK(musaGetLastError());
}
