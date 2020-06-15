#include "parrots_cpp_helper.hpp"

void CARAFEForwardCUDAKernelLauncher(
    const DArrayLite features, const DArrayLite masks, DArrayLite rfeatures,
    DArrayLite routput, DArrayLite rmasks, DArrayLite output,
    const int kernel_size, const int group_size, const int scale_factor,
    cudaStream_t stream);

void CARAFEBackwardCUDAKernelLauncher(
    const DArrayLite top_grad, const DArrayLite rfeatures,
    const DArrayLite masks, DArrayLite rtop_grad, DArrayLite rbottom_grad_hs,
    DArrayLite rbottom_grad, DArrayLite rmask_grad, DArrayLite bottom_grad,
    DArrayLite mask_grad, const int kernel_size, const int group_size,
    const int scale_factor, cudaStream_t stream);

void carafe_forward_cuda(CudaContext& ctx, const SSElement& attr,
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

  auto& rfeatures = outs[0];
  auto& routput = outs[1];
  auto& rmasks = outs[2];
  auto& output = outs[3];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  CARAFEForwardCUDAKernelLauncher(features, masks, rfeatures, routput, rmasks,
                                  output, kernel_size, group_size, scale_factor,
                                  stream);
}

void carafe_backward_cuda(CudaContext& ctx, const SSElement& attr,
                          const OperatorBase::in_list_t& ins,
                          OperatorBase::out_list_t& outs) {
  int kernel_size, group_size, scale_factor;
  SSAttrs(attr)
      .get<int>("kernel_size", kernel_size)
      .get<int>("group_size", group_size)
      .get<int>("scale_factor", scale_factor)
      .done();

  const auto& top_grad = ins[0];
  const auto& rfeatures = ins[1];
  const auto& masks = ins[2];

  auto& rtop_grad = outs[0];
  auto rbottom_grad_hs = outs[1];
  auto& rbottom_grad = outs[2];
  auto& rmask_grad = outs[3];
  auto& bottom_grad = outs[4];
  auto& mask_grad = outs[5];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  CARAFEBackwardCUDAKernelLauncher(top_grad, rfeatures, masks, rtop_grad,
                                   rbottom_grad_hs, rbottom_grad, rmask_grad,
                                   bottom_grad, mask_grad, kernel_size,
                                   group_size, scale_factor, stream);
}

PARROTS_EXTENSION_REGISTER(carafe_forward)
    .attr("kernel_size")
    .attr("group_size")
    .attr("scale_factor")
    .input(2)
    .output(4)
    .apply(carafe_forward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(carafe_backward)
    .attr("kernel_size")
    .attr("group_size")
    .attr("scale_factor")
    .input(3)
    .output(6)
    .apply(carafe_backward_cuda)
    .done();
