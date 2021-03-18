#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void SyncBNForwardMeanCUDAKernelLauncher(const Tensor input, Tensor mean);

void SyncBNForwardVarCUDAKernelLauncher(const Tensor input, const Tensor mean,
                                        Tensor var);

void SyncBNForwardOutputCUDAKernelLauncher(
    const Tensor input, const Tensor mean, const Tensor var,
    Tensor running_mean, Tensor running_var, const Tensor weight,
    const Tensor bias, Tensor norm, Tensor std, Tensor output, float eps,
    float momentum, int group_size);

void SyncBNBackwardParamCUDAKernelLauncher(const Tensor grad_output,
                                           const Tensor norm,
                                           Tensor grad_weight,
                                           Tensor grad_bias);

void SyncBNBackwardDataCUDAKernelLauncher(const Tensor grad_output,
                                          const Tensor weight,
                                          const Tensor grad_weight,
                                          const Tensor grad_bias,
                                          const Tensor norm, const Tensor std,
                                          Tensor grad_input);

void sync_bn_forward_mean_cuda(const Tensor input, Tensor mean) {
  SyncBNForwardMeanCUDAKernelLauncher(input, mean);
}

void sync_bn_forward_var_cuda(const Tensor input, const Tensor mean,
                              Tensor var) {
  SyncBNForwardVarCUDAKernelLauncher(input, mean, var);
}

void sync_bn_forward_output_cuda(const Tensor input, const Tensor mean,
                                 const Tensor var, Tensor running_mean,
                                 Tensor running_var, const Tensor weight,
                                 const Tensor bias, Tensor norm, Tensor std,
                                 Tensor output, float eps, float momentum,
                                 int group_size) {
  SyncBNForwardOutputCUDAKernelLauncher(input, mean, var, running_mean,
                                        running_var, weight, bias, norm, std,
                                        output, eps, momentum, group_size);
}

void sync_bn_backward_param_cuda(const Tensor grad_output, const Tensor norm,
                                 Tensor grad_weight, Tensor grad_bias) {
  SyncBNBackwardParamCUDAKernelLauncher(grad_output, norm, grad_weight,
                                        grad_bias);
}

void sync_bn_backward_data_cuda(const Tensor grad_output, const Tensor weight,
                                const Tensor grad_weight,
                                const Tensor grad_bias, const Tensor norm,
                                const Tensor std, Tensor grad_input) {
  SyncBNBackwardDataCUDAKernelLauncher(grad_output, weight, grad_weight,
                                       grad_bias, norm, std, grad_input);
}
#endif

void sync_bn_forward_mean(const Tensor input, Tensor mean) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(mean);
    sync_bn_forward_mean_cuda(input, mean);
#else
    AT_ERROR("SyncBatchNorm is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("SyncBatchNorm is not implemented on CPU");
  }
}

void sync_bn_forward_var(const Tensor input, const Tensor mean, Tensor var) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(mean);
    CHECK_CUDA_INPUT(var);
    sync_bn_forward_var_cuda(input, mean, var);
#else
    AT_ERROR("SyncBatchNorm is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("SyncBatchNorm is not implemented on CPU");
  }
}

void sync_bn_forward_output(const Tensor input, const Tensor mean,
                            const Tensor var, const Tensor weight,
                            const Tensor bias, Tensor running_mean,
                            Tensor running_var, Tensor norm, Tensor std,
                            Tensor output, float eps, float momentum,
                            int group_size) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(mean);
    CHECK_CUDA_INPUT(var);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(bias);
    CHECK_CUDA_INPUT(running_mean);
    CHECK_CUDA_INPUT(running_var);
    CHECK_CUDA_INPUT(norm);
    CHECK_CUDA_INPUT(std);
    CHECK_CUDA_INPUT(output);
    sync_bn_forward_output_cuda(input, mean, var, running_mean, running_var,
                                weight, bias, norm, std, output, eps, momentum,
                                group_size);
#else
    AT_ERROR("SyncBatchNorm is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("SyncBatchNorm is not implemented on CPU");
  }
}

void sync_bn_backward_param(const Tensor grad_output, const Tensor norm,
                            Tensor grad_weight, Tensor grad_bias) {
  if (grad_output.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(grad_output);
    CHECK_CUDA_INPUT(norm);
    CHECK_CUDA_INPUT(grad_weight);
    CHECK_CUDA_INPUT(grad_bias);
    sync_bn_backward_param_cuda(grad_output, norm, grad_weight, grad_bias);
#else
    AT_ERROR("SyncBatchNorm is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("SyncBatchNorm is not implemented on CPU");
  }
}

void sync_bn_backward_data(const Tensor grad_output, const Tensor weight,
                           const Tensor grad_weight, const Tensor grad_bias,
                           const Tensor norm, const Tensor std,
                           Tensor grad_input) {
  if (grad_output.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(grad_output);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(grad_weight);
    CHECK_CUDA_INPUT(grad_bias);
    CHECK_CUDA_INPUT(norm);
    CHECK_CUDA_INPUT(std);
    CHECK_CUDA_INPUT(grad_input);
    sync_bn_backward_data_cuda(grad_output, weight, grad_weight, grad_bias,
                               norm, std, grad_input);
#else
    AT_ERROR("SyncBatchNorm is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("SyncBatchNorm is not implemented on CPU");
  }
}
