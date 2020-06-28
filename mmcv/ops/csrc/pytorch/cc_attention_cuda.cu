// Modified from
// https://github.com/LikeLy-Journey/SegmenTron/blob/master/segmentron/modules/csrc/criss_cross_attention/ca_cuda.cu

#include <THC/THC.h>

#include <THC/THCDeviceUtils.cuh>

#include "cc_attention_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

void CAForwardCUDAKernelLauncher(const Tensor t, const Tensor f,
                                 Tensor weight) {
  AT_ASSERTM(t.device().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(f.device().is_cuda(), "input must be a CUDA tensor");

  auto n = t.size(0);
  auto c = t.size(1);
  auto h = t.size(2);
  auto w = t.size(3);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Run kernel
  dim3 threads(32, 32);
  int d1 = (w + threads.x - 1) / threads.x;
  int d2 = (h + threads.y - 1) / threads.y;
  int d3 = h + w;
  dim3 blocks(d1, d2, d3);

  AT_DISPATCH_FLOATING_TYPES(t.scalar_type(), "ca_forward", [&] {
    ca_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        t.contiguous().data_ptr<scalar_t>(),
        f.contiguous().data_ptr<scalar_t>(),
        weight.contiguous().data_ptr<scalar_t>(), n, c, h, w);
  });
  THCudaCheck(cudaGetLastError());
}

void CABackwardCUDAKernelLauncher(const Tensor dw, const Tensor t,
                                  const Tensor f, Tensor dt, Tensor df) {
  AT_ASSERTM(dw.device().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(t.device().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(f.device().is_cuda(), "input must be a CUDA tensor");

  auto n = t.size(0);
  auto c = t.size(1);
  auto h = t.size(2);
  auto w = t.size(3);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Run kernel
  dim3 threads(32, 32);
  int d1 = (w + threads.x - 1) / threads.x;
  int d2 = (h + threads.y - 1) / threads.y;
  int d3 = c;
  dim3 blocks(d1, d2, d3);

  AT_DISPATCH_FLOATING_TYPES(t.scalar_type(), "ca_backward_kernel_t", [&] {
    ca_backward_kernel_t<scalar_t><<<blocks, threads, 0, stream>>>(
        dw.contiguous().data_ptr<scalar_t>(),
        t.contiguous().data_ptr<scalar_t>(),
        f.contiguous().data_ptr<scalar_t>(),
        dt.contiguous().data_ptr<scalar_t>(), n, c, h, w);
  });

  AT_DISPATCH_FLOATING_TYPES(f.scalar_type(), "ca_backward_kernel_f", [&] {
    ca_backward_kernel_f<scalar_t><<<blocks, threads, 0, stream>>>(
        dw.contiguous().data_ptr<scalar_t>(),
        t.contiguous().data_ptr<scalar_t>(),
        f.contiguous().data_ptr<scalar_t>(),
        df.contiguous().data_ptr<scalar_t>(), n, c, h, w);
  });
  THCudaCheck(cudaGetLastError());
}

void CAMapForwardCUDAKernelLauncher(const Tensor weight, const Tensor g,
                                    Tensor out) {
  AT_ASSERTM(weight.device().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(g.device().is_cuda(), "input must be a CUDA tensor");

  auto n = g.size(0);
  auto c = g.size(1);
  auto h = g.size(2);
  auto w = g.size(3);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Run kernel
  dim3 threads(32, 32);
  int d1 = (w + threads.x - 1) / threads.x;
  int d2 = (h + threads.y - 1) / threads.y;
  int d3 = c;
  dim3 blocks(d1, d2, d3);

  AT_DISPATCH_FLOATING_TYPES(g.scalar_type(), "ca_map_forward", [&] {
    ca_map_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        weight.contiguous().data_ptr<scalar_t>(),
        g.contiguous().data_ptr<scalar_t>(),
        out.contiguous().data_ptr<scalar_t>(), n, c, h, w);
  });
  THCudaCheck(cudaGetLastError());
}

void CAMapBackwardCUDAKernelLauncher(const Tensor dout, const Tensor weight,
                                     const Tensor g, Tensor dw, Tensor dg) {
  AT_ASSERTM(dout.device().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(weight.device().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(g.device().is_cuda(), "input must be a CUDA tensor");

  auto n = dout.size(0);
  auto c = dout.size(1);
  auto h = dout.size(2);
  auto w = dout.size(3);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Run kernel
  dim3 threads(32, 32);
  int d1 = (w + threads.x - 1) / threads.x;
  int d2 = (h + threads.y - 1) / threads.y;
  int d3 = h + w;
  dim3 blocks(d1, d2, d3);

  AT_DISPATCH_FLOATING_TYPES(
      weight.scalar_type(), "ca_map_backward_kernel_w", [&] {
        ca_map_backward_kernel_w<scalar_t><<<blocks, threads, 0, stream>>>(
            dout.contiguous().data_ptr<scalar_t>(),
            weight.contiguous().data_ptr<scalar_t>(),
            g.contiguous().data_ptr<scalar_t>(),
            dw.contiguous().data_ptr<scalar_t>(), n, c, h, w);
      });

  AT_DISPATCH_FLOATING_TYPES(g.scalar_type(), "ca_map_backward_kernel_g", [&] {
    ca_map_backward_kernel_g<scalar_t><<<blocks, threads, 0, stream>>>(
        dout.contiguous().data_ptr<scalar_t>(),
        weight.contiguous().data_ptr<scalar_t>(),
        g.contiguous().data_ptr<scalar_t>(),
        dg.contiguous().data_ptr<scalar_t>(), n, c, h, w);
  });
  THCudaCheck(cudaGetLastError());
}
