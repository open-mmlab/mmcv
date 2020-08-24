#include "parrots_cpp_helper.hpp"

void CAForwardCUDAKernelLauncher(const DArrayLite t, const DArrayLite f,
                                 DArrayLite weight, CudaContext &ctx,
                                 cudaStream_t stream);

void CABackwardCUDAKernelLauncher(const DArrayLite dw, const DArrayLite t,
                                  const DArrayLite f, DArrayLite dt,
                                  DArrayLite df, CudaContext &ctx,
                                  cudaStream_t stream);

void CAMapForwardCUDAKernelLauncher(const DArrayLite weight, const DArrayLite g,
                                    DArrayLite out, CudaContext &ctx,
                                    cudaStream_t stream);

void CAMapBackwardCUDAKernelLauncher(const DArrayLite dout,
                                     const DArrayLite weight,
                                     const DArrayLite g, DArrayLite dw,
                                     DArrayLite dg, CudaContext &ctx,
                                     cudaStream_t stream);

void ca_forward_cuda(CudaContext &ctx, const SSElement &attr,
                     const OperatorBase::in_list_t &ins,
                     OperatorBase::out_list_t &outs) {
  const auto &t = ins[0];
  const auto &f = ins[1];
  auto &weight = outs[0];
  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  CAForwardCUDAKernelLauncher(t, f, weight, ctx, stream);
}

void ca_backward_cuda(CudaContext &ctx, const SSElement &attr,
                      const OperatorBase::in_list_t &ins,
                      OperatorBase::out_list_t &outs) {
  const auto &dw = ins[0];
  const auto &t = ins[1];
  const auto &f = ins[2];
  auto &dt = outs[0];
  auto &df = outs[1];
  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  CABackwardCUDAKernelLauncher(dw, t, f, dt, df, ctx, stream);
}

void ca_map_forward_cuda(CudaContext &ctx, const SSElement &attr,
                         const OperatorBase::in_list_t &ins,
                         OperatorBase::out_list_t &outs) {
  const auto &weight = ins[0];
  const auto &g = ins[1];
  auto &out = outs[0];
  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  CAMapForwardCUDAKernelLauncher(weight, g, out, ctx, stream);
}

void ca_map_backward_cuda(CudaContext &ctx, const SSElement &attr,
                          const OperatorBase::in_list_t &ins,
                          OperatorBase::out_list_t &outs) {
  const auto &dout = ins[0];
  const auto &weight = ins[1];
  const auto &g = ins[2];
  auto &dw = outs[0];
  auto &dg = outs[1];
  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  CAMapBackwardCUDAKernelLauncher(dout, weight, g, dw, dg, ctx, stream);
}

PARROTS_EXTENSION_REGISTER(ca_forward)
    .input(2)
    .output(1)
    .apply(ca_forward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(ca_backward)
    .input(3)
    .output(2)
    .apply(ca_backward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(ca_map_forward)
    .input(2)
    .output(1)
    .apply(ca_map_forward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(ca_map_backward)
    .input(3)
    .output(2)
    .apply(ca_map_backward_cuda)
    .done();
