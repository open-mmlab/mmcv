#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void ROIAlignRotatedForwardCUDAKernelLauncher(const at::Tensor features,
                                              const at::Tensor rois,
                                              const float spatial_scale,
                                              const int sample_num,
                                              const bool aligned,
                                              const int channels,
                                              const int height,
                                              const int width,
                                              const int num_rois,
                                              const int pooled_height,
                                              const int pooled_width,
                                              at::Tensor output);

void ROIAlignRotatedBackwardCUDAKernelLauncher(const at::Tensor top_grad,
                                               const at::Tensor rois,
                                               const float spatial_scale,
                                               const int sample_num,
                                               const bool aligned,
                                               const int channels,
                                               const int height,
                                               const int width,
                                               const int num_rois,
                                               const int pooled_height,
                                               const int pooled_width,
                                               at::Tensor bottom_grad);

void roi_align_rotated_forward_cuda(Tensor features, Tensor rois,
                                    Tensor output,
                                    int pooled_height, int pooled_width,
                                    float spatial_scale, int sample_num, bool aligned) {
  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);

  if (size_rois != 6) {
    AT_ERROR("wrong roi size");
  }

  int num_channels = features.size(1);
  int data_height = features.size(2);
  int data_width = features.size(3);
  ROIAlignRotatedForwardCUDAKernelLauncher(features, rois, spatial_scale,
                                           sample_num, aligned, num_channels,
                                           data_height, data_width, num_rois,
                                           pooled_height, pooled_width,
                                           output);
}

void roi_align_rotated_backward_cuda(Tensor top_grad, Tensor rois,
                                     Tensor bottom_grad,
                                     int pooled_height, int pooled_width,
                                     float spatial_scale, int sample_num,
                                     bool aligned) {
  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);
  if (size_rois != 6) {
    AT_ERROR("wrong roi size");
  }

  int num_channels = bottom_grad.size(1);
  int data_height = bottom_grad.size(2);
  int data_width = bottom_grad.size(3);
  ROIAlignRotatedBackwardCUDAKernelLauncher(top_grad, rois, spatial_scale,
                                            sample_num, aligned, num_channels,
                                            data_height, data_width, num_rois,
                                            pooled_height, pooled_width,
                                            bottom_grad);
}
#endif


void roi_align_rotated_forward(Tensor input, Tensor rois, Tensor output,
                               int pooled_height,
                               int pooled_width, float spatial_scale,
                               int sample_num, bool aligned) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(rois);
    CHECK_CUDA_INPUT(output);

    roi_align_rotated_forward_cuda(input, rois, output,
                                   pooled_height, pooled_width, spatial_scale,
                                   sample_num, aligned);
#else
    AT_ERROR("RoIAlignRotated is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("RoIAlignRotated is not supported for cpu");
  }
}

void roi_align_rotated_backward(Tensor grad_output, Tensor rois,
                                Tensor grad_input, int pooled_height,
                                int pooled_width, float spatial_scale,
                                int sample_num, bool aligned) {
  if (grad_output.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(grad_output);
    CHECK_CUDA_INPUT(rois);
    CHECK_CUDA_INPUT(grad_input);

    roi_align_rotated_backward_cuda(grad_output, rois, grad_input,
                                    pooled_height, pooled_width, spatial_scale,
                                    sample_num, aligned);
#else
    AT_ERROR("RoIAlignRotated is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("RoIAlignRotated is not supported for cpu");
  }
}
