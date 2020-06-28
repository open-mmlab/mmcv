#include "cc_attention_cuda_kernel.cuh"
#include "parrots_cuda_helper.hpp"

void CAForwardCUDAKernelLauncher(const DArrayLite t, const DArrayLite f,
                                 DArrayLite weight, CudaContext &ctx,
                                 cudaStream_t stream) {
  auto n = t.dim(0);
  auto c = t.dim(1);
  auto h = t.dim(2);
  auto w = t.dim(3);

  // Run kernel
  dim3 threads(32, 32);
  int d1 = (w + threads.x - 1) / threads.x;
  int d2 = (h + threads.y - 1) / threads.y;
  int d3 = h + w;
  dim3 blocks(d1, d2, d3);

  PARROTS_DISPATCH_FLOATING_TYPES(t.elemType().prim(), [&] {
    ca_forward_kernel<scalar_t>
        <<<blocks, threads, 0, stream>>>(t.ptr<scalar_t>(), f.ptr<scalar_t>(),
                                         weight.ptr<scalar_t>(), n, c, h, w);
  });
  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void CABackwardCUDAKernelLauncher(const DArrayLite dw, const DArrayLite t,
                                  const DArrayLite f, DArrayLite dt,
                                  DArrayLite df, CudaContext &ctx,
                                  cudaStream_t stream) {
  auto n = t.dim(0);
  auto c = t.dim(1);
  auto h = t.dim(2);
  auto w = t.dim(3);

  // Run kernel
  dim3 threads(32, 32);
  int d1 = (w + threads.x - 1) / threads.x;
  int d2 = (h + threads.y - 1) / threads.y;
  int d3 = c;
  dim3 blocks(d1, d2, d3);

  PARROTS_DISPATCH_FLOATING_TYPES(t.elemType().prim(), [&] {
    ca_backward_kernel_t<scalar_t><<<blocks, threads, 0, stream>>>(
        dw.ptr<scalar_t>(), t.ptr<scalar_t>(), f.ptr<scalar_t>(),
        dt.ptr<scalar_t>(), n, c, h, w);
  });

  PARROTS_DISPATCH_FLOATING_TYPES(f.elemType().prim(), [&] {
    ca_backward_kernel_f<scalar_t><<<blocks, threads, 0, stream>>>(
        dw.ptr<scalar_t>(), t.ptr<scalar_t>(), f.ptr<scalar_t>(),
        df.ptr<scalar_t>(), n, c, h, w);
  });
  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void CAMapForwardCUDAKernelLauncher(const DArrayLite weight, const DArrayLite g,
                                    DArrayLite out, CudaContext &ctx,
                                    cudaStream_t stream) {
  auto n = g.dim(0);
  auto c = g.dim(1);
  auto h = g.dim(2);
  auto w = g.dim(3);

  // Run kernel
  dim3 threads(32, 32);
  int d1 = (w + threads.x - 1) / threads.x;
  int d2 = (h + threads.y - 1) / threads.y;
  int d3 = c;
  dim3 blocks(d1, d2, d3);

  PARROTS_DISPATCH_FLOATING_TYPES(g.elemType().prim(), [&] {
    ca_map_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        weight.ptr<scalar_t>(), g.ptr<scalar_t>(), out.ptr<scalar_t>(), n, c, h,
        w);
  });
  PARROTS_CUDA_CHECK(cudaGetLastError());
}

void CAMapBackwardCUDAKernelLauncher(const DArrayLite dout,
                                     const DArrayLite weight,
                                     const DArrayLite g, DArrayLite dw,
                                     DArrayLite dg, CudaContext &ctx,
                                     cudaStream_t stream) {
  auto n = dout.dim(0);
  auto c = dout.dim(1);
  auto h = dout.dim(2);
  auto w = dout.dim(3);

  // Run kernel
  dim3 threads(32, 32);
  int d1 = (w + threads.x - 1) / threads.x;
  int d2 = (h + threads.y - 1) / threads.y;
  int d3 = h + w;
  dim3 blocks(d1, d2, d3);

  PARROTS_DISPATCH_FLOATING_TYPES(weight.elemType().prim(), [&] {
    ca_map_backward_kernel_w<scalar_t><<<blocks, threads, 0, stream>>>(
        dout.ptr<scalar_t>(), weight.ptr<scalar_t>(), g.ptr<scalar_t>(),
        dw.ptr<scalar_t>(), n, c, h, w);
  });

  PARROTS_DISPATCH_FLOATING_TYPES(g.elemType().prim(), [&] {
    ca_map_backward_kernel_g<scalar_t><<<blocks, threads, 0, stream>>>(
        dout.ptr<scalar_t>(), weight.ptr<scalar_t>(), g.ptr<scalar_t>(),
        dg.ptr<scalar_t>(), n, c, h, w);
  });
  PARROTS_CUDA_CHECK(cudaGetLastError());
}
