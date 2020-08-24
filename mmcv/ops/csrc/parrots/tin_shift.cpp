#include "parrots_cpp_helper.hpp"

void TINShiftForwardCUDAKernelLauncher(const DArrayLite input,
                                       const DArrayLite shift,
                                       DArrayLite output, cudaStream_t stream);

void TINShiftBackwardCUDAKernelLauncher(const DArrayLite grad_output,
                                        const DArrayLite shift,
                                        DArrayLite grad_input,
                                        cudaStream_t stream);

void tin_shift_forward_cuda(CudaContext &ctx, const SSElement &attr,
                            const OperatorBase::in_list_t &ins,
                            OperatorBase::out_list_t &outs) {
  const auto &input = ins[0];
  const auto &shift = ins[1];
  auto &output = outs[0];
  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  TINShiftForwardCUDAKernelLauncher(input, shift, output, stream);
}

void tin_shift_backward_cuda(CudaContext &ctx, const SSElement &attr,
                             const OperatorBase::in_list_t &ins,
                             OperatorBase::out_list_t &outs) {
  const auto &grad_output = ins[0];
  const auto &shift = ins[1];
  auto &grad_input = outs[0];
  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  TINShiftBackwardCUDAKernelLauncher(grad_output, shift, grad_input, stream);
}

PARROTS_EXTENSION_REGISTER(tin_shift_forward)
    .input(2)
    .output(1)
    .apply(tin_shift_forward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(tin_shift_backward)
    .input(2)
    .output(1)
    .apply(tin_shift_backward_cuda)
    .done();
