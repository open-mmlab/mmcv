// Copyright (c) OpenMMLab. All rights reserved
#ifndef ROI_ALIGN_ROTATED_PYTORCH_H
#define ROI_ALIGN_ROTATED_PYTORCH_H
#include <torch/types.h>
using namespace at;

#ifdef MMCV_WITH_CUDA
void roi_align_rotated_forward_cuda(Tensor features, Tensor rois, Tensor output,
                                    int pooled_height, int pooled_width,
                                    float spatial_scale, int sample_num,
                                    bool aligned, bool clockwise);

void roi_align_rotated_backward_cuda(Tensor grad_output, Tensor rois,
                                     Tensor bottom_grad, int pooled_height,
                                     int pooled_width, float spatial_scale,
                                     int sample_num, bool aligned,
                                     bool clockwise);
#endif

void roi_align_rotated_forward_cpu(Tensor features, Tensor rois, Tensor output,
                                   int pooled_height, int pooled_width,
                                   float spatial_scale, int sample_num,
                                   bool aligned, bool clockwise);

void roi_align_rotated_backward_cpu(Tensor grad_output, Tensor rois,
                                    Tensor bottom_grad, int pooled_height,
                                    int pooled_width, float spatial_scale,
                                    int sample_num, bool aligned,
                                    bool clockwise);

#endif  // ROI_ALIGN_ROTATED_PYTORCH_H
