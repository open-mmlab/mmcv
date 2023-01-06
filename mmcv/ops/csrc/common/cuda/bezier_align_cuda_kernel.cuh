// Copyright (c) OpenMMLab. All rights reserved
// Modified from
// https://github.com/aim-uofa/AdelaiDet/blob/master/adet/layers/csrc/BezierAlign/BezierAlign_cuda.cu
#ifndef BEZIER_ALIGN_CUDA_KERNEL_CUH
#define BEZIER_ALIGN_CUDA_KERNEL_CUH

#include <float.h>
#ifdef MMCV_WITH_TRT
#include "common_cuda_helper.hpp"
#else  // MMCV_WITH_TRT
#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else  // MMCV_USE_PARROTS
#include "pytorch_cuda_helper.hpp"
#endif  // MMCV_USE_PARROTS
#endif  // MMCV_WITH_TRT

template <typename T>
__device__ T bezier_curve(const T p0, const T p1, const T p2, const T p3,
                          const T u) {
  return ((1. - u) * (1. - u) * (1. - u) * p0 +
          3. * u * (1. - u) * (1. - u) * p1 + 3. * u * u * (1. - u) * p2 +
          u * u * u * p3);
}

template <typename T>
__global__ void bezier_align_forward_cuda_kernel(
    const int nthreads,
    const T *bottom_data,  // inputs
    const T *bottom_rois,  // bottom rois contains the bezier curve
    T *top_data,           // outputs
    const int pooled_height, const int pooled_width, const T spatial_scale,
    const int sampling_ratio, bool aligned, const int channels,
    const int height, const int width) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    // beziers have size Nx(1+8*2) = Nx17
    const T *offset_bottom_rois = bottom_rois + n * 17;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not use rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;

    // TODO: avoid this by using parallel annotation, for good
    T p0_x = offset_bottom_rois[1] * spatial_scale;
    T p0_y = offset_bottom_rois[2] * spatial_scale;
    T p1_x = offset_bottom_rois[3] * spatial_scale;
    T p1_y = offset_bottom_rois[4] * spatial_scale;
    T p2_x = offset_bottom_rois[5] * spatial_scale;
    T p2_y = offset_bottom_rois[6] * spatial_scale;
    T p3_x = offset_bottom_rois[7] * spatial_scale;
    T p3_y = offset_bottom_rois[8] * spatial_scale;
    T p4_x = offset_bottom_rois[15] * spatial_scale;
    T p4_y = offset_bottom_rois[16] * spatial_scale;
    T p5_x = offset_bottom_rois[13] * spatial_scale;
    T p5_y = offset_bottom_rois[14] * spatial_scale;
    T p6_x = offset_bottom_rois[11] * spatial_scale;
    T p6_y = offset_bottom_rois[12] * spatial_scale;
    T p7_x = offset_bottom_rois[9] * spatial_scale;
    T p7_y = offset_bottom_rois[10] * spatial_scale;

    // compute the coords
    const T u = pw / static_cast<T>(pooled_width);
    const T v = ph / static_cast<T>(pooled_height);
    const T x0 = bezier_curve(p0_x, p1_x, p2_x, p3_x, u);
    const T y0 = bezier_curve(p0_y, p1_y, p2_y, p3_y, u);
    const T x1 = bezier_curve(p4_x, p5_x, p6_x, p7_x, u);
    const T y1 = bezier_curve(p4_y, p5_y, p6_y, p7_y, u);
    const T x_center = x1 * v + x0 * (1. - v) - offset;
    const T y_center = y1 * v + y0 * (1. - v) - offset;

    T roi_width = max(abs(p0_x - p3_x), abs(p4_x - p7_x));
    T roi_height = max(abs(p0_y - p3_y), abs(p4_y - p7_y));
    if (!aligned) {  // for backward-compatibility only
      roi_width = max(roi_width, (T)1.);
      roi_height = max(roi_height, (T)1.);
    }
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T *offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    // When the grid is empty, output zeros == 0/1, instead of NaN.
    const T count = max(roi_bin_grid_h * roi_bin_grid_w, 1);  // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++)  // e.g., iy = 0, 1
    {
      const T y = y_center - (T)0.5 * bin_size_h +
                  static_cast<T>(iy + .5f) * bin_size_h /
                      static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = x_center - (T)0.5 * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate(offset_bottom_data, height, width, y, x,
                                     index);
        output_val += val;
      }
    }
    output_val /= count;

    top_data[index] = output_val;
  }
}

template <typename T>
__global__ void bezier_align_backward_cuda_kernel(
    const int nthreads, const T *top_diff, const T *bottom_rois, T *bottom_diff,
    const int pooled_height, const int pooled_width, const T spatial_scale,
    const int sampling_ratio, bool aligned, const int channels,
    const int height, const int width) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    // beziers have size Nx(1+8*2) = Nx17
    const T *offset_bottom_rois = bottom_rois + n * 17;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not use rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;
    T p0_x = offset_bottom_rois[1] * spatial_scale;
    T p0_y = offset_bottom_rois[2] * spatial_scale;
    T p1_x = offset_bottom_rois[3] * spatial_scale;
    T p1_y = offset_bottom_rois[4] * spatial_scale;
    T p2_x = offset_bottom_rois[5] * spatial_scale;
    T p2_y = offset_bottom_rois[6] * spatial_scale;
    T p3_x = offset_bottom_rois[7] * spatial_scale;
    T p3_y = offset_bottom_rois[8] * spatial_scale;
    T p4_x = offset_bottom_rois[15] * spatial_scale;
    T p4_y = offset_bottom_rois[16] * spatial_scale;
    T p5_x = offset_bottom_rois[13] * spatial_scale;
    T p5_y = offset_bottom_rois[14] * spatial_scale;
    T p6_x = offset_bottom_rois[11] * spatial_scale;
    T p6_y = offset_bottom_rois[12] * spatial_scale;
    T p7_x = offset_bottom_rois[9] * spatial_scale;
    T p7_y = offset_bottom_rois[10] * spatial_scale;

    // compute the coords
    const T u = pw / static_cast<T>(pooled_width);
    const T v = ph / static_cast<T>(pooled_height);
    const T x0 = bezier_curve(p0_x, p1_x, p2_x, p3_x, u);
    const T y0 = bezier_curve(p0_y, p1_y, p2_y, p3_y, u);
    const T x1 = bezier_curve(p4_x, p5_x, p6_x, p7_x, u);
    const T y1 = bezier_curve(p4_y, p5_y, p6_y, p7_y, u);
    const T x_center = x1 * v + x0 * (1. - v) - offset;
    const T y_center = y1 * v + y0 * (1. - v) - offset;

    T roi_width = max(abs(p0_x - p3_x), abs(p4_x - p7_x));
    T roi_height = max(abs(p0_y - p3_y), abs(p4_y - p7_y));
    if (!aligned) {  // for backward-compatibility only
      roi_width = max(roi_width, (T)1.);
      roi_height = max(roi_height, (T)1.);
    }
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    T *offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const T *offset_top_diff = top_diff + top_offset;
    const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++)  // e.g., iy = 0, 1
    {
      const T y = y_center - (T)0.5 * bin_size_h +
                  static_cast<T>(iy + .5f) * bin_size_h /
                      static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = x_center - (T)0.5 * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4,
                                      x_low, x_high, y_low, y_high, index);

        T g1 = top_diff_this_bin * w1 / count;
        T g2 = top_diff_this_bin * w2 / count;
        T g3 = top_diff_this_bin * w3 / count;
        T g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(offset_bottom_diff + y_low * width + x_low,
                    static_cast<T>(g1));
          atomicAdd(offset_bottom_diff + y_low * width + x_high,
                    static_cast<T>(g2));
          atomicAdd(offset_bottom_diff + y_high * width + x_low,
                    static_cast<T>(g3));
          atomicAdd(offset_bottom_diff + y_high * width + x_high,
                    static_cast<T>(g4));
        }  // if
      }    // ix
    }      // iy
  }        // CUDA_1D_KERNEL_LOOP
}  // BezierAlignBackward

#endif  // BEZIER_ALIGN_CUDA_KERNEL_CUH
