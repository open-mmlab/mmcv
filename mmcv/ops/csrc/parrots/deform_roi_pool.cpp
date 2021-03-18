#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void DeformRoIPoolForwardCUDAKernelLauncher(Tensor input, Tensor rois,
                                            Tensor offset, Tensor output,
                                            int pooled_height, int pooled_width,
                                            float spatial_scale,
                                            int sampling_ratio, float gamma);

void DeformRoIPoolBackwardCUDAKernelLauncher(
    Tensor grad_output, Tensor input, Tensor rois, Tensor offset,
    Tensor grad_input, Tensor grad_offset, int pooled_height, int pooled_width,
    float spatial_scale, int sampling_ratio, float gamma);

void deform_roi_pool_forward_cuda(Tensor input, Tensor rois, Tensor offset,
                                  Tensor output, int pooled_height,
                                  int pooled_width, float spatial_scale,
                                  int sampling_ratio, float gamma) {
  DeformRoIPoolForwardCUDAKernelLauncher(input, rois, offset, output,
                                         pooled_height, pooled_width,
                                         spatial_scale, sampling_ratio, gamma);
}

void deform_roi_pool_backward_cuda(Tensor grad_output, Tensor input,
                                   Tensor rois, Tensor offset,
                                   Tensor grad_input, Tensor grad_offset,
                                   int pooled_height, int pooled_width,
                                   float spatial_scale, int sampling_ratio,
                                   float gamma) {
  DeformRoIPoolBackwardCUDAKernelLauncher(
      grad_output, input, rois, offset, grad_input, grad_offset, pooled_height,
      pooled_width, spatial_scale, sampling_ratio, gamma);
}
#endif

void deform_roi_pool_forward(Tensor input, Tensor rois, Tensor offset,
                             Tensor output, int pooled_height, int pooled_width,
                             float spatial_scale, int sampling_ratio,
                             float gamma) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(rois);
    CHECK_CUDA_INPUT(offset);
    CHECK_CUDA_INPUT(output);

    deform_roi_pool_forward_cuda(input, rois, offset, output, pooled_height,
                                 pooled_width, spatial_scale, sampling_ratio,
                                 gamma);
#else
    AT_ERROR("DeformRoIPool is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("DeformRoIPool is not implemented on CPU");
  }
}

void deform_roi_pool_backward(Tensor grad_output, Tensor input, Tensor rois,
                              Tensor offset, Tensor grad_input,
                              Tensor grad_offset, int pooled_height,
                              int pooled_width, float spatial_scale,
                              int sampling_ratio, float gamma) {
  if (grad_output.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(grad_output);
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(rois);
    CHECK_CUDA_INPUT(offset);
    CHECK_CUDA_INPUT(grad_input);
    CHECK_CUDA_INPUT(grad_offset);

    deform_roi_pool_backward_cuda(grad_output, input, rois, offset, grad_input,
                                  grad_offset, pooled_height, pooled_width,
                                  spatial_scale, sampling_ratio, gamma);
#else
    AT_ERROR("DeformRoIPool is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("DeformRoIPool is not implemented on CPU");
  }
}
