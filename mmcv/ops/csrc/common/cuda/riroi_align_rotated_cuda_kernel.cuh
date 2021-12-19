// Modified from
// https://github.com/csuhan/ReDet/blob/master/mmdet/ops/riroi_align/src/riroi_align_kernel.cu
#ifndef RIROI_ALIGN_ROTATED_CUDA_KERNEL_CUH
#define RIROI_ALIGN_ROTATED_CUDA_KERNEL_CUH

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

/*** Forward ***/
template <typename scalar_t>
__global__ void riroi_align_rotated_forward_cuda_kernel(
    const int nthreads, const scalar_t *bottom_data,
    const scalar_t *bottom_rois, const scalar_t spatial_scale,
    const int sample_num, const bool clockwise, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int nOrientation, scalar_t *top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int o = (index / pooled_width / pooled_height) % nOrientation;
    int c = (index / pooled_width / pooled_height / nOrientation) % channels;
    int n = index / pooled_width / pooled_height / nOrientation / channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    scalar_t roi_center_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_center_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_width = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_height = offset_bottom_rois[4] * spatial_scale;
    // scalar_t theta = offset_bottom_rois[5] * M_PI / 180.0;
    scalar_t theta = offset_bottom_rois[5];
    if (clockwise) {
      theta = -theta;  // If clockwise, the angle needs to be reversed.
    }
    // Force malformed ROIs to be 1x1
    roi_width = max(roi_width, (scalar_t)1.);
    roi_height = max(roi_height, (scalar_t)1.);
    scalar_t bin_size_h = static_cast<scalar_t>(roi_height) /
                          static_cast<scalar_t>(pooled_height);
    scalar_t bin_size_w =
        static_cast<scalar_t>(roi_width) / static_cast<scalar_t>(pooled_width);

    // find aligned index
    scalar_t ind_float = theta * nOrientation / (2 * M_PI);
    int ind = floor(ind_float);
    scalar_t l_var = ind_float - (scalar_t)ind;
    scalar_t r_var = 1.0 - l_var;
    // correct start channel
    ind = (ind + nOrientation) % nOrientation;
    // rotated channel
    int ind_rot = (o - ind + nOrientation) % nOrientation;
    int ind_rot_plus = (ind_rot + 1 + nOrientation) % nOrientation;
    const scalar_t *offset_bottom_data =
        bottom_data +
        (roi_batch_ind * channels * nOrientation + c * nOrientation + ind_rot) *
            height * width;

    const scalar_t *offset_bottom_data_plus =
        bottom_data + (roi_batch_ind * channels * nOrientation +
                       c * nOrientation + ind_rot_plus) *
                          height * width;
    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sample_num > 0)
                             ? sample_num
                             : ceilf(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sample_num > 0) ? sample_num : ceilf(roi_width / pooled_width);

    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    scalar_t roi_start_h = -roi_height / 2.0;
    scalar_t roi_start_w = -roi_width / 2.0;
    scalar_t cosscalar_theta = cos(theta);
    scalar_t sinscalar_theta = sin(theta);

    // We do average (integral) pooling inside a bin
    const scalar_t count = max(roi_bin_grid_h * roi_bin_grid_w, 1);  // e.g. = 4

    scalar_t output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
      const scalar_t yy =
          roi_start_h + ph * bin_size_h +
          static_cast<scalar_t>(iy + .5f) * bin_size_h /
              static_cast<scalar_t>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const scalar_t xx = roi_start_w + pw * bin_size_w +
                            static_cast<scalar_t>(ix + .5f) * bin_size_w /
                                static_cast<scalar_t>(roi_bin_grid_w);

        // Rotate by theta (counterclockwise) around the center and translate
        scalar_t y = yy * cosscalar_theta - xx * sinscalar_theta + roi_center_h;
        scalar_t x = yy * sinscalar_theta + xx * cosscalar_theta + roi_center_w;

        scalar_t val = bilinear_interpolate<scalar_t>(
            offset_bottom_data, height, width, y, x, index);
        scalar_t val_plus = bilinear_interpolate<scalar_t>(
            offset_bottom_data_plus, height, width, y, x, index);
        output_val += r_var * val + l_var * val_plus;
      }
    }
    output_val /= count;

    top_data[index] = output_val;
  }
}

/*** Backward ***/
template <typename scalar_t>
__global__ void riroi_align_rotated_backward_cuda_kernel(
    const int nthreads, const scalar_t *top_diff, const scalar_t *bottom_rois,
    const scalar_t spatial_scale, const int sample_num, const bool clockwise,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int nOrientation,
    scalar_t *bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int o = (index / pooled_width / pooled_height) % nOrientation;
    int c = (index / pooled_width / pooled_height / nOrientation) % channels;
    int n = index / pooled_width / pooled_height / nOrientation / channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not round
    scalar_t roi_center_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_center_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_width = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_height = offset_bottom_rois[4] * spatial_scale;
    // scalar_t theta = offset_bottom_rois[5] * M_PI / 180.0;
    scalar_t theta = offset_bottom_rois[5];
    if (clockwise) {
      theta = -theta;  // If clockwise, the angle needs to be reversed.
    }
    // Force malformed ROIs to be 1x1
    roi_width = max(roi_width, (scalar_t)1.);
    roi_height = max(roi_height, (scalar_t)1.);

    scalar_t bin_size_h = static_cast<scalar_t>(roi_height) /
                          static_cast<scalar_t>(pooled_height);
    scalar_t bin_size_w =
        static_cast<scalar_t>(roi_width) / static_cast<scalar_t>(pooled_width);

    // find aligned index
    scalar_t ind_float = theta * nOrientation / (2 * M_PI);
    int ind = floor(ind_float);
    scalar_t l_var = ind_float - (scalar_t)ind;
    scalar_t r_var = 1.0 - l_var;
    // correct start channel
    ind = (ind + nOrientation) % nOrientation;
    // rotated channel
    int ind_rot = (o - ind + nOrientation) % nOrientation;
    int ind_rot_plus = (ind_rot + 1 + nOrientation) % nOrientation;
    scalar_t *offset_bottom_diff =
        bottom_diff +
        (roi_batch_ind * channels * nOrientation + c * nOrientation + ind_rot) *
            height * width;
    scalar_t *offset_bottom_diff_plus =
        bottom_diff + (roi_batch_ind * channels * nOrientation +
                       c * nOrientation + ind_rot_plus) *
                          height * width;
    int top_offset = (n * channels * nOrientation + c * nOrientation + o) *
                     pooled_height * pooled_width;
    const scalar_t *offset_top_diff = top_diff + top_offset;
    const scalar_t top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sample_num > 0)
                             ? sample_num
                             : ceilf(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sample_num > 0) ? sample_num : ceilf(roi_width / pooled_width);

    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    scalar_t roi_start_h = -roi_height / 2.0;
    scalar_t roi_start_w = -roi_width / 2.0;
    scalar_t cosTheta = cos(theta);
    scalar_t sinTheta = sin(theta);

    // We do average (integral) pooling inside a bin
    const scalar_t count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
      const scalar_t yy =
          roi_start_h + ph * bin_size_h +
          static_cast<scalar_t>(iy + .5f) * bin_size_h /
              static_cast<scalar_t>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const scalar_t xx = roi_start_w + pw * bin_size_w +
                            static_cast<scalar_t>(ix + .5f) * bin_size_w /
                                static_cast<scalar_t>(roi_bin_grid_w);

        // Rotate by theta around the center and translate
        scalar_t y = yy * cosTheta - xx * sinTheta + roi_center_h;
        scalar_t x = yy * sinTheta + xx * cosTheta + roi_center_w;

        scalar_t w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient<scalar_t>(height, width, y, x, w1, w2, w3,
                                                w4, x_low, x_high, y_low,
                                                y_high, index);

        scalar_t g1 = top_diff_this_bin * w1 / count;
        scalar_t g2 = top_diff_this_bin * w2 / count;
        scalar_t g3 = top_diff_this_bin * w3 / count;
        scalar_t g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(offset_bottom_diff + y_low * width + x_low, g1 * r_var);
          atomicAdd(offset_bottom_diff + y_low * width + x_high, g2 * r_var);
          atomicAdd(offset_bottom_diff + y_high * width + x_low, g3 * r_var);
          atomicAdd(offset_bottom_diff + y_high * width + x_high, g4 * r_var);

          atomicAdd(offset_bottom_diff_plus + y_low * width + x_low,
                    g1 * l_var);
          atomicAdd(offset_bottom_diff_plus + y_low * width + x_high,
                    g2 * l_var);
          atomicAdd(offset_bottom_diff_plus + y_high * width + x_low,
                    g3 * l_var);
          atomicAdd(offset_bottom_diff_plus + y_high * width + x_high,
                    g4 * l_var);

        }  // if
      }    // ix
    }      // iy
  }        // CUDA_1D_KERNEL_LOOP
}  // RiRoIAlignBackward

#endif  // RIROI_ALIGN_ROTATED_CUDA_KERNEL_CUH
