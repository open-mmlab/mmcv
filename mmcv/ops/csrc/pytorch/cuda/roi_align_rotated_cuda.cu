// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cuda_helper.hpp"
#include "pytorch_device_registry.hpp"
#include "roi_align_rotated_cuda_kernel.cuh"

void ROIAlignRotatedForwardCUDAKernelLauncher(
    const at::Tensor features, const at::Tensor rois, const float spatial_scale,
    const int sample_num, const bool aligned, const bool clockwise,
    const int channels, const int height, const int width, const int num_rois,
    const int pooled_height, const int pooled_width, at::Tensor output) {
  const int output_size = num_rois * pooled_height * pooled_width * channels;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.scalar_type(), "ROIAlignRotatedLaucherForward", ([&] {
        const scalar_t *bottom_data = features.data_ptr<scalar_t>();
        const scalar_t *rois_data = rois.data_ptr<scalar_t>();
        scalar_t *top_data = output.data_ptr<scalar_t>();

        roi_align_rotated_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, bottom_data, rois_data, scalar_t(spatial_scale),
                sample_num, aligned, clockwise, channels, height, width,
                pooled_height, pooled_width, top_data);
      }));

  AT_CUDA_CHECK(cudaGetLastError());
}

void ROIAlignRotatedBackwardCUDAKernelLauncher(
    const at::Tensor top_grad, const at::Tensor rois, const float spatial_scale,
    const int sample_num, const bool aligned, const bool clockwise,
    const int channels, const int height, const int width, const int num_rois,
    const int pooled_height, const int pooled_width, at::Tensor bottom_grad) {
  const int output_size = num_rois * pooled_height * pooled_width * channels;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.scalar_type(), "ROIAlignLaucherBackward", ([&] {
        const scalar_t *top_diff = top_grad.data_ptr<scalar_t>();
        const scalar_t *rois_data = rois.data_ptr<scalar_t>();
        scalar_t *bottom_diff = bottom_grad.data_ptr<scalar_t>();
        roi_align_rotated_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, top_diff, rois_data, spatial_scale, sample_num,
                aligned, clockwise, channels, height, width, pooled_height,
                pooled_width, bottom_diff);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

void roi_align_rotated_forward_cuda(Tensor features, Tensor rois, Tensor output,
                                    int aligned_height, int aligned_width,
                                    float spatial_scale, int sample_ratio,
                                    bool aligned, bool clockwise) {
  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);

  if (size_rois != 6) {
    AT_ERROR("wrong roi size");
  }

  int num_channels = features.size(1);
  int data_height = features.size(2);
  int data_width = features.size(3);
  ROIAlignRotatedForwardCUDAKernelLauncher(
      features, rois, spatial_scale, sample_ratio, aligned, clockwise,
      num_channels, data_height, data_width, num_rois, aligned_height,
      aligned_width, output);
}

void roi_align_rotated_backward_cuda(Tensor top_grad, Tensor rois,
                                     Tensor bottom_grad, int aligned_height,
                                     int aligned_width, float spatial_scale,
                                     int sample_ratio, bool aligned,
                                     bool clockwise) {
  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);
  if (size_rois != 6) {
    AT_ERROR("wrong roi size");
  }

  int num_channels = bottom_grad.size(1);
  int data_height = bottom_grad.size(2);
  int data_width = bottom_grad.size(3);
  ROIAlignRotatedBackwardCUDAKernelLauncher(
      top_grad, rois, spatial_scale, sample_ratio, aligned, clockwise,
      num_channels, data_height, data_width, num_rois, aligned_height,
      aligned_width, bottom_grad);
}

void roi_align_rotated_forward_impl(Tensor features, Tensor rois, Tensor output,
                                    int aligned_height, int aligned_width,
                                    float spatial_scale, int sample_ratio,
                                    bool aligned, bool clockwise);

void roi_align_rotated_backward_impl(Tensor top_grad, Tensor rois,
                                     Tensor bottom_grad, int aligned_height,
                                     int aligned_width, float spatial_scale,
                                     int sample_ratio, bool aligned,
                                     bool clockwise);
REGISTER_DEVICE_IMPL(roi_align_rotated_forward_impl, CUDA,
                     roi_align_rotated_forward_cuda);
REGISTER_DEVICE_IMPL(roi_align_rotated_backward_impl, CUDA,
                     roi_align_rotated_backward_cuda);
