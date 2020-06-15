#include "parrots_cpp_helper.hpp"

void SyncBNForwardMeanCUDAKernelLauncher(const DArrayLite input,
                                         DArrayLite mean, cudaStream_t stream);

void SyncBNForwardVarCUDAKernelLauncher(const DArrayLite input,
                                        const DArrayLite mean, DArrayLite var,
                                        cudaStream_t stream);

void SyncBNForwardOutputCUDAKernelLauncher(
    const DArrayLite input, const DArrayLite mean, const DArrayLite var,
    DArrayLite running_mean, DArrayLite running_var, const DArrayLite weight,
    const DArrayLite bias, DArrayLite norm, DArrayLite std, DArrayLite output,
    const float eps, const float momentum, size_t group_size,
    cudaStream_t stream);

void SyncBNBackwardParamCUDAKernelLauncher(const DArrayLite grad_output,
                                           const DArrayLite norm,
                                           DArrayLite weight_diff,
                                           DArrayLite bias_diff,
                                           cudaStream_t stream);

void SyncBNBackwardDataCUDAKernelLauncher(
    const DArrayLite grad_output, const DArrayLite weight,
    const DArrayLite weight_diff, const DArrayLite bias_diff,
    const DArrayLite norm, const DArrayLite std, DArrayLite grad_input,
    cudaStream_t stream);

void sync_bn_forward_mean_cuda(CudaContext& ctx, const SSElement& attr,
                               const OperatorBase::in_list_t& ins,
                               OperatorBase::out_list_t& outs) {
  const auto& input = ins[0];
  auto& mean = outs[0];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  SyncBNForwardMeanCUDAKernelLauncher(input, mean, stream);
}

void sync_bn_forward_var_cuda(CudaContext& ctx, const SSElement& attr,
                              const OperatorBase::in_list_t& ins,
                              OperatorBase::out_list_t& outs) {
  const auto& input = ins[0];
  const auto& mean = ins[1];
  auto& var = outs[0];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  SyncBNForwardVarCUDAKernelLauncher(input, mean, var, stream);
}

void sync_bn_forward_output_cuda(CudaContext& ctx, const SSElement& attr,
                                 const OperatorBase::in_list_t& ins,
                                 OperatorBase::out_list_t& outs) {
  size_t group_size;
  float eps, momentum;
  SSAttrs(attr)
      .get<float>("eps", eps)
      .get<float>("momentum", momentum)
      .get<size_t>("group_size", group_size)
      .done();

  const auto& input = ins[0];
  const auto& mean = ins[1];
  const auto& var = ins[2];
  const auto& weight = ins[3];
  const auto& bias = ins[4];
  auto& running_mean = outs[0];
  auto& running_var = outs[1];
  auto& norm = outs[2];
  auto& std = outs[3];
  auto& output = outs[4];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  SyncBNForwardOutputCUDAKernelLauncher(
      input, mean, var, running_mean, running_var, weight, bias, norm, std,
      output, eps, momentum, group_size, stream);
}

void sync_bn_backward_param_cuda(CudaContext& ctx, const SSElement& attr,
                                 const OperatorBase::in_list_t& ins,
                                 OperatorBase::out_list_t& outs) {
  const auto& grad_output = ins[0];
  const auto& norm = ins[1];
  auto& grad_weight = outs[0];
  auto& grad_bias = outs[1];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  SyncBNBackwardParamCUDAKernelLauncher(grad_output, norm, grad_weight,
                                        grad_bias, stream);
}

void sync_bn_backward_data_cuda(CudaContext& ctx, const SSElement& attr,
                                const OperatorBase::in_list_t& ins,
                                OperatorBase::out_list_t& outs) {
  const auto& grad_output = ins[0];
  const auto& weight = ins[1];
  const auto& grad_weight = ins[2];
  const auto& grad_bias = ins[3];
  const auto& norm = ins[4];
  const auto& std = ins[5];
  auto& grad_input = outs[0];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  SyncBNBackwardDataCUDAKernelLauncher(grad_output, weight, grad_weight,
                                       grad_bias, norm, std, grad_input,
                                       stream);
}

PARROTS_EXTENSION_REGISTER(sync_bn_forward_mean)
    .input(1)
    .output(1)
    .apply(sync_bn_forward_mean_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(sync_bn_forward_var)
    .input(2)
    .output(1)
    .apply(sync_bn_forward_var_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(sync_bn_forward_output)
    .attr("eps")
    .attr("momentum")
    .attr("group_size")
    .input(5)
    .output(5)
    .apply(sync_bn_forward_output_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(sync_bn_backward_param)
    .input(2)
    .output(2)
    .apply(sync_bn_backward_param_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(sync_bn_backward_data)
    .input(6)
    .output(1)
    .apply(sync_bn_backward_data_cuda)
    .done();
