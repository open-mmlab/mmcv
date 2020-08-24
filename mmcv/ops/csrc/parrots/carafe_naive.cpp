#include "parrots_cpp_helper.hpp"

void CARAFENAIVEForwardCUDAKernelLauncher(
    const DArrayLite features, const DArrayLite masks, DArrayLite output,
    const int kernel_size, const int group_size, const int scale_factor,
    cudaStream_t stream);

void CARAFENAIVEBackwardCUDAKernelLauncher(
    const DArrayLite top_grad, const DArrayLite features,
    const DArrayLite masks, DArrayLite bottom_grad, DArrayLite mask_grad,
    const int kernel_size, const int group_size, const int scale_factor,
    cudaStream_t stream);

void carafe_naive_forward_cuda(CudaContext& ctx, const SSElement& attr,
                               const OperatorBase::in_list_t& ins,
                               OperatorBase::out_list_t& outs) {
  int kernel_size, group_size, scale_factor;
  SSAttrs(attr)
      .get<int>("kernel_size", kernel_size)
      .get<int>("group_size", group_size)
      .get<int>("scale_factor", scale_factor)
      .done();

  const auto& features = ins[0];
  const auto& masks = ins[1];

  auto& output = outs[0];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  CARAFENAIVEForwardCUDAKernelLauncher(features, masks, output, kernel_size,
                                       group_size, scale_factor, stream);
}

void carafe_naive_backward_cuda(CudaContext& ctx, const SSElement& attr,
                                const OperatorBase::in_list_t& ins,
                                OperatorBase::out_list_t& outs) {
  int kernel_size, group_size, scale_factor;
  SSAttrs(attr)
      .get<int>("kernel_size", kernel_size)
      .get<int>("group_size", group_size)
      .get<int>("scale_factor", scale_factor)
      .done();

  const auto& top_grad = ins[0];
  const auto& features = ins[1];
  const auto& masks = ins[2];

  auto& bottom_grad = outs[0];
  auto& mask_grad = outs[1];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  CARAFENAIVEBackwardCUDAKernelLauncher(top_grad, features, masks, bottom_grad,
                                        mask_grad, kernel_size, group_size,
                                        scale_factor, stream);
}

PARROTS_EXTENSION_REGISTER(carafe_naive_forward)
    .attr("kernel_size")
    .attr("group_size")
    .attr("scale_factor")
    .input(2)
    .output(1)
    .apply(carafe_naive_forward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(carafe_naive_backward)
    .attr("kernel_size")
    .attr("group_size")
    .attr("scale_factor")
    .input(3)
    .output(2)
    .apply(carafe_naive_backward_cuda)
    .done();
