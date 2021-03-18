#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void CARAFEForwardCUDAKernelLauncher(const Tensor features, const Tensor masks,
                                     Tensor rfeatures, Tensor routput,
                                     Tensor rmasks, Tensor output,
                                     const int kernel_size,
                                     const int group_size,
                                     const int scale_factor);

void CARAFEBackwardCUDAKernelLauncher(
    const Tensor top_grad, const Tensor rfeatures, const Tensor masks,
    Tensor rtop_grad, Tensor rbottom_grad_hs, Tensor rbottom_grad,
    Tensor rmask_grad, Tensor bottom_grad, Tensor mask_grad,
    const int kernel_size, const int group_size, const int scale_factor);

void carafe_forward_cuda(Tensor features, Tensor masks, Tensor rfeatures,
                         Tensor routput, Tensor rmasks, Tensor output,
                         int kernel_size, int group_size, int scale_factor) {
  CARAFEForwardCUDAKernelLauncher(features, masks, rfeatures, routput, rmasks,
                                  output, kernel_size, group_size,
                                  scale_factor);
}

void carafe_backward_cuda(Tensor top_grad, Tensor rfeatures, Tensor masks,
                          Tensor rtop_grad, Tensor rbottom_grad_hs,
                          Tensor rbottom_grad, Tensor rmask_grad,
                          Tensor bottom_grad, Tensor mask_grad, int kernel_size,
                          int group_size, int scale_factor) {
  CARAFEBackwardCUDAKernelLauncher(top_grad, rfeatures, masks, rtop_grad,
                                   rbottom_grad_hs, rbottom_grad, rmask_grad,
                                   bottom_grad, mask_grad, kernel_size,
                                   group_size, scale_factor);
}
#endif

void carafe_forward(Tensor features, Tensor masks, Tensor rfeatures,
                    Tensor routput, Tensor rmasks, Tensor output,
                    int kernel_size, int group_size, int scale_factor) {
  if (features.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(features);
    CHECK_CUDA_INPUT(masks);
    CHECK_CUDA_INPUT(rfeatures);
    CHECK_CUDA_INPUT(routput);
    CHECK_CUDA_INPUT(rmasks);
    CHECK_CUDA_INPUT(output);
    carafe_forward_cuda(features, masks, rfeatures, routput, rmasks, output,
                        kernel_size, group_size, scale_factor);
#else
    AT_ERROR("Carafe is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("Carafe is not implemented on CPU");
  }
}

void carafe_backward(Tensor top_grad, Tensor rfeatures, Tensor masks,
                     Tensor rtop_grad, Tensor rbottom_grad_hs,
                     Tensor rbottom_grad, Tensor rmask_grad, Tensor bottom_grad,
                     Tensor mask_grad, int kernel_size, int group_size,
                     int scale_factor) {
  if (top_grad.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(top_grad);
    CHECK_CUDA_INPUT(rfeatures);
    CHECK_CUDA_INPUT(masks);
    CHECK_CUDA_INPUT(rtop_grad);
    CHECK_CUDA_INPUT(rbottom_grad_hs);
    CHECK_CUDA_INPUT(rbottom_grad);
    CHECK_CUDA_INPUT(rmask_grad);
    CHECK_CUDA_INPUT(bottom_grad);
    CHECK_CUDA_INPUT(mask_grad);
    carafe_backward_cuda(top_grad, rfeatures, masks, rtop_grad, rbottom_grad_hs,
                         rbottom_grad, rmask_grad, bottom_grad, mask_grad,
                         kernel_size, group_size, scale_factor);
#else
    AT_ERROR("Carafe is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("Carafe is not implemented on CPU");
  }
}
