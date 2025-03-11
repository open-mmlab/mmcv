#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
void AssignScoreWithKForwardMUSAKernelLauncher(
    int B, int N0, int N1, int M, int K, int O, int aggregate,
    const Tensor &points, const Tensor &centers, const Tensor &scores,
    const Tensor &knn_idx, Tensor &output);

void AssignScoreWithKBackwardMUSAKernelLauncher(
    int B, int N0, int N1, int M, int K, int O, int aggregate,
    const Tensor &grad_out, const Tensor &points, const Tensor &centers,
    const Tensor &scores, const Tensor &knn_idx, Tensor &grad_points,
    Tensor &grad_centers, Tensor &grad_scores);

void assign_score_withk_forward_musa(int B, int N0, int N1, int M, int K, int O,
                                     int aggregate, const Tensor &points,
                                     const Tensor &centers,
                                     const Tensor &scores,
                                     const Tensor &knn_idx, Tensor &output) {
  AssignScoreWithKForwardMUSAKernelLauncher(
      B, N0, N1, M, K, O, aggregate, points, centers, scores, knn_idx, output);
};

void assign_score_withk_backward_musa(
    int B, int N0, int N1, int M, int K, int O, int aggregate,
    const Tensor &grad_out, const Tensor &points, const Tensor &centers,
    const Tensor &scores, const Tensor &knn_idx, Tensor &grad_points,
    Tensor &grad_centers, Tensor &grad_scores) {
  AssignScoreWithKBackwardMUSAKernelLauncher(
      B, N0, N1, M, K, O, aggregate, grad_out, points, centers, scores, knn_idx,
      grad_points, grad_centers, grad_scores);
};

void assign_score_withk_forward_impl(int B, int N0, int N1, int M, int K, int O,
                                     int aggregate, const Tensor &points,
                                     const Tensor &centers,
                                     const Tensor &scores,
                                     const Tensor &knn_idx, Tensor &output);

void assign_score_withk_backward_impl(
    int B, int N0, int N1, int M, int K, int O, int aggregate,
    const Tensor &grad_out, const Tensor &points, const Tensor &centers,
    const Tensor &scores, const Tensor &knn_idx, Tensor &grad_points,
    Tensor &grad_centers, Tensor &grad_scores);

REGISTER_DEVICE_IMPL(assign_score_withk_forward_impl, MUSA,
                     assign_score_withk_forward_musa);
REGISTER_DEVICE_IMPL(assign_score_withk_backward_impl, MUSA,
                     assign_score_withk_backward_musa);

void BallQueryForwardMUSAKernelLauncher(int b, int n, int m, float min_radius,
                                        float max_radius, int nsample,
                                        const Tensor new_xyz, const Tensor xyz,
                                        Tensor idx);

void ball_query_forward_musa(int b, int n, int m, float min_radius,
                             float max_radius, int nsample,
                             const Tensor new_xyz, const Tensor xyz,
                             Tensor idx) {
  BallQueryForwardMUSAKernelLauncher(b, n, m, min_radius, max_radius, nsample,
                                     new_xyz, xyz, idx);
};

void ball_query_forward_impl(int b, int n, int m, float min_radius,
                             float max_radius, int nsample,
                             const Tensor new_xyz, const Tensor xyz,
                             Tensor idx);
REGISTER_DEVICE_IMPL(ball_query_forward_impl, MUSA, ball_query_forward_musa);

void StackBallQueryForwardMUSAKernelLauncher(float max_radius, int nsample,
                                             const Tensor new_xyz,
                                             const Tensor new_xyz_batch_cnt,
                                             const Tensor xyz,
                                             const Tensor xyz_batch_cnt,
                                             Tensor idx);

void stack_ball_query_forward_musa(float max_radius, int nsample,
                                   const Tensor new_xyz,
                                   const Tensor new_xyz_batch_cnt,
                                   const Tensor xyz, const Tensor xyz_batch_cnt,
                                   Tensor idx) {
  StackBallQueryForwardMUSAKernelLauncher(
      max_radius, nsample, new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx);
};

void stack_ball_query_forward_impl(float max_radius, int nsample,
                                   const Tensor new_xyz,
                                   const Tensor new_xyz_batch_cnt,
                                   const Tensor xyz, const Tensor xyz_batch_cnt,
                                   Tensor idx);
REGISTER_DEVICE_IMPL(stack_ball_query_forward_impl, MUSA,
                     stack_ball_query_forward_musa);

void BBoxOverlapsMUSAKernelLauncher(const Tensor bboxes1, const Tensor bboxes2,
                                    Tensor ious, const int mode,
                                    const bool aligned, const int offset);

void bbox_overlaps_musa(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                        const int mode, const bool aligned, const int offset) {
  BBoxOverlapsMUSAKernelLauncher(bboxes1, bboxes2, ious, mode, aligned, offset);
}

void bbox_overlaps_impl(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                        const int mode, const bool aligned, const int offset);
REGISTER_DEVICE_IMPL(bbox_overlaps_impl, MUSA, bbox_overlaps_musa);

void BorderAlignForwardMUSAKernelLauncher(const Tensor &input,
                                          const Tensor &boxes, Tensor output,
                                          Tensor argmax_idx,
                                          const int pool_size);

void BorderAlignBackwardMUSAKernelLauncher(const Tensor &grad_output,
                                           const Tensor &boxes,
                                           const Tensor &argmax_idx,
                                           Tensor grad_input,
                                           const int pool_size);

void border_align_forward_musa(const Tensor &input, const Tensor &boxes,
                               Tensor output, Tensor argmax_idx,
                               const int pool_size) {
  BorderAlignForwardMUSAKernelLauncher(input, boxes, output, argmax_idx,
                                       pool_size);
}

void border_align_backward_musa(const Tensor &grad_output, const Tensor &boxes,
                                const Tensor &argmax_idx, Tensor grad_input,
                                const int pool_size) {
  BorderAlignBackwardMUSAKernelLauncher(grad_output, boxes, argmax_idx,
                                        grad_input, pool_size);
}

void border_align_forward_impl(const Tensor &input, const Tensor &boxes,
                               Tensor output, Tensor argmax_idx,
                               const int pool_size);

void border_align_backward_impl(const Tensor &grad_output, const Tensor &boxes,
                                const Tensor &argmax_idx, Tensor grad_input,
                                const int pool_size);

REGISTER_DEVICE_IMPL(border_align_forward_impl, MUSA,
                     border_align_forward_musa);
REGISTER_DEVICE_IMPL(border_align_backward_impl, MUSA,
                     border_align_backward_musa);

void box_iou_rotated_musa(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                          const int mode_flag, const bool aligned);

void box_iou_rotated_impl(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                          const int mode_flag, const bool aligned);
REGISTER_DEVICE_IMPL(box_iou_rotated_impl, MUSA, box_iou_rotated_musa);

void box_iou_quadri_musa(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                         const int mode_flag, const bool aligned);

void box_iou_quadri_impl(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                         const int mode_flag, const bool aligned);
REGISTER_DEVICE_IMPL(box_iou_quadri_impl, MUSA, box_iou_quadri_musa);

#if ((!defined(MUSA_ARCH)) || (defined(MUSA_ARCH)) && (MUSA_ARCH > 21))

void CARAFEForwardMUSAKernelLauncher(const Tensor features, const Tensor masks,
                                     Tensor rfeatures, Tensor routput,
                                     Tensor rmasks, Tensor output,
                                     const int kernel_size,
                                     const int group_size,
                                     const int scale_factor);

void CARAFEBackwardMUSAKernelLauncher(
    const Tensor top_grad, const Tensor rfeatures, const Tensor masks,
    Tensor rtop_grad, Tensor rbottom_grad_hs, Tensor rbottom_grad,
    Tensor rmask_grad, Tensor bottom_grad, Tensor mask_grad,
    const int kernel_size, const int group_size, const int scale_factor);

void carafe_forward_musa(Tensor features, Tensor masks, Tensor rfeatures,
                         Tensor routput, Tensor rmasks, Tensor output,
                         int kernel_size, int group_size, int scale_factor) {
  CARAFEForwardMUSAKernelLauncher(features, masks, rfeatures, routput, rmasks,
                                  output, kernel_size, group_size,
                                  scale_factor);
}

void carafe_backward_musa(Tensor top_grad, Tensor rfeatures, Tensor masks,
                          Tensor rtop_grad, Tensor rbottom_grad_hs,
                          Tensor rbottom_grad, Tensor rmask_grad,
                          Tensor bottom_grad, Tensor mask_grad, int kernel_size,
                          int group_size, int scale_factor) {
  CARAFEBackwardMUSAKernelLauncher(top_grad, rfeatures, masks, rtop_grad,
                                   rbottom_grad_hs, rbottom_grad, rmask_grad,
                                   bottom_grad, mask_grad, kernel_size,
                                   group_size, scale_factor);
}

void carafe_forward_impl(Tensor features, Tensor masks, Tensor rfeatures,
                         Tensor routput, Tensor rmasks, Tensor output,
                         int kernel_size, int group_size, int scale_factor);

void carafe_backward_impl(Tensor top_grad, Tensor rfeatures, Tensor masks,
                          Tensor rtop_grad, Tensor rbottom_grad_hs,
                          Tensor rbottom_grad, Tensor rmask_grad,
                          Tensor bottom_grad, Tensor mask_grad, int kernel_size,
                          int group_size, int scale_factor);

REGISTER_DEVICE_IMPL(carafe_forward_impl, MUSA, carafe_forward_musa);
REGISTER_DEVICE_IMPL(carafe_backward_impl, MUSA, carafe_backward_musa);
#endif

void CARAFENAIVEForwardMUSAKernelLauncher(const Tensor features,
                                          const Tensor masks, Tensor output,
                                          const int kernel_size,
                                          const int group_size,
                                          const int scale_factor);

void CARAFENAIVEBackwardMUSAKernelLauncher(
    const Tensor top_grad, const Tensor features, const Tensor masks,
    Tensor bottom_grad, Tensor mask_grad, const int kernel_size,
    const int group_size, const int scale_factor);

void carafe_naive_forward_musa(Tensor features, Tensor masks, Tensor output,
                               int kernel_size, int group_size,
                               int scale_factor) {
  CARAFENAIVEForwardMUSAKernelLauncher(features, masks, output, kernel_size,
                                       group_size, scale_factor);
}

void carafe_naive_backward_musa(Tensor top_grad, Tensor features, Tensor masks,
                                Tensor bottom_grad, Tensor mask_grad,
                                int kernel_size, int group_size,
                                int scale_factor) {
  CARAFENAIVEBackwardMUSAKernelLauncher(top_grad, features, masks, bottom_grad,
                                        mask_grad, kernel_size, group_size,
                                        scale_factor);
}
void carafe_naive_forward_impl(Tensor features, Tensor masks, Tensor output,
                               int kernel_size, int group_size,
                               int scale_factor);

void carafe_naive_backward_impl(Tensor top_grad, Tensor features, Tensor masks,
                                Tensor bottom_grad, Tensor mask_grad,
                                int kernel_size, int group_size,
                                int scale_factor);

REGISTER_DEVICE_IMPL(carafe_naive_forward_impl, MUSA,
                     carafe_naive_forward_musa);
REGISTER_DEVICE_IMPL(carafe_naive_backward_impl, MUSA,
                     carafe_naive_backward_musa);

void CorrelationForwardMUSAKernelLauncher(Tensor input1, Tensor input2,
                                          Tensor output, int kH, int kW,
                                          int patchH, int patchW, int padH,
                                          int padW, int dilationH,
                                          int dilationW, int dilation_patchH,
                                          int dilation_patchW, int dH, int dW);

void CorrelationBackwardMUSAKernelLauncher(Tensor grad_output, Tensor input1,
                                           Tensor input2, Tensor grad_input1,
                                           Tensor grad_input2, int kH, int kW,
                                           int patchH, int patchW, int padH,
                                           int padW, int dilationH,
                                           int dilationW, int dilation_patchH,
                                           int dilation_patchW, int dH, int dW);

void correlation_forward_musa(Tensor input1, Tensor input2, Tensor output,
                              int kH, int kW, int patchH, int patchW, int padH,
                              int padW, int dilationH, int dilationW,
                              int dilation_patchH, int dilation_patchW, int dH,
                              int dW) {
  CorrelationForwardMUSAKernelLauncher(
      input1, input2, output, kH, kW, patchH, patchW, padH, padW, dilationH,
      dilationW, dilation_patchH, dilation_patchW, dH, dW);
}

void correlation_backward_musa(Tensor grad_output, Tensor input1, Tensor input2,
                               Tensor grad_input1, Tensor grad_input2, int kH,
                               int kW, int patchH, int patchW, int padH,
                               int padW, int dilationH, int dilationW,
                               int dilation_patchH, int dilation_patchW, int dH,
                               int dW) {
  CorrelationBackwardMUSAKernelLauncher(
      grad_output, input1, input2, grad_input1, grad_input2, kH, kW, patchH,
      patchW, padH, padW, dilationH, dilationW, dilation_patchH,
      dilation_patchW, dH, dW);
}

void correlation_forward_impl(Tensor input1, Tensor input2, Tensor output,
                              int kH, int kW, int patchH, int patchW, int padH,
                              int padW, int dilationH, int dilationW,
                              int dilation_patchH, int dilation_patchW, int dH,
                              int dW);

void correlation_backward_impl(Tensor grad_output, Tensor input1, Tensor input2,
                               Tensor grad_input1, Tensor grad_input2, int kH,
                               int kW, int patchH, int patchW, int padH,
                               int padW, int dilationH, int dilationW,
                               int dilation_patchH, int dilation_patchW, int dH,
                               int dW);

REGISTER_DEVICE_IMPL(correlation_forward_impl, MUSA, correlation_forward_musa);
REGISTER_DEVICE_IMPL(correlation_backward_impl, MUSA,
                     correlation_backward_musa);

void deformable_im2col_musa(Tensor data_im, Tensor data_offset,
                            const int channels, const int height,
                            const int width, const int ksize_h,
                            const int ksize_w, const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w,
                            const int dilation_h, const int dilation_w,
                            const int parallel_imgs, const int deformable_group,
                            Tensor data_col);

void deformable_col2im_musa(Tensor data_col, Tensor data_offset,
                            const int channels, const int height,
                            const int width, const int ksize_h,
                            const int ksize_w, const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w,
                            const int dilation_h, const int dilation_w,
                            const int parallel_imgs, const int deformable_group,
                            Tensor grad_im);

void deformable_col2im_coord_musa(
    Tensor data_col, Tensor data_im, Tensor data_offset, const int channels,
    const int height, const int width, const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int parallel_imgs,
    const int deformable_group, Tensor grad_offset);

void deformable_im2col_impl(Tensor data_im, Tensor data_offset,
                            const int channels, const int height,
                            const int width, const int ksize_h,
                            const int ksize_w, const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w,
                            const int dilation_h, const int dilation_w,
                            const int parallel_imgs, const int deformable_group,
                            Tensor data_col);

void deformable_col2im_impl(Tensor data_col, Tensor data_offset,
                            const int channels, const int height,
                            const int width, const int ksize_h,
                            const int ksize_w, const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w,
                            const int dilation_h, const int dilation_w,
                            const int parallel_imgs, const int deformable_group,
                            Tensor grad_im);

void deformable_col2im_coord_impl(
    Tensor data_col, Tensor data_im, Tensor data_offset, const int channels,
    const int height, const int width, const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int parallel_imgs,
    const int deformable_group, Tensor grad_offset);

REGISTER_DEVICE_IMPL(deformable_im2col_impl, MUSA, deformable_im2col_musa);
REGISTER_DEVICE_IMPL(deformable_col2im_impl, MUSA, deformable_col2im_musa);
REGISTER_DEVICE_IMPL(deformable_col2im_coord_impl, MUSA,
                     deformable_col2im_coord_musa);

void DeformRoIPoolForwardMUSAKernelLauncher(Tensor input, Tensor rois,
                                            Tensor offset, Tensor output,
                                            int pooled_height, int pooled_width,
                                            float spatial_scale,
                                            int sampling_ratio, float gamma);

void DeformRoIPoolBackwardMUSAKernelLauncher(
    Tensor grad_output, Tensor input, Tensor rois, Tensor offset,
    Tensor grad_input, Tensor grad_offset, int pooled_height, int pooled_width,
    float spatial_scale, int sampling_ratio, float gamma);

void deform_roi_pool_forward_musa(Tensor input, Tensor rois, Tensor offset,
                                  Tensor output, int pooled_height,
                                  int pooled_width, float spatial_scale,
                                  int sampling_ratio, float gamma) {
  DeformRoIPoolForwardMUSAKernelLauncher(input, rois, offset, output,
                                         pooled_height, pooled_width,
                                         spatial_scale, sampling_ratio, gamma);
}

void deform_roi_pool_backward_musa(Tensor grad_output, Tensor input,
                                   Tensor rois, Tensor offset,
                                   Tensor grad_input, Tensor grad_offset,
                                   int pooled_height, int pooled_width,
                                   float spatial_scale, int sampling_ratio,
                                   float gamma) {
  DeformRoIPoolBackwardMUSAKernelLauncher(
      grad_output, input, rois, offset, grad_input, grad_offset, pooled_height,
      pooled_width, spatial_scale, sampling_ratio, gamma);
}

void deform_roi_pool_forward_impl(Tensor input, Tensor rois, Tensor offset,
                                  Tensor output, int pooled_height,
                                  int pooled_width, float spatial_scale,
                                  int sampling_ratio, float gamma);

void deform_roi_pool_backward_impl(Tensor grad_output, Tensor input,
                                   Tensor rois, Tensor offset,
                                   Tensor grad_input, Tensor grad_offset,
                                   int pooled_height, int pooled_width,
                                   float spatial_scale, int sampling_ratio,
                                   float gamma);

REGISTER_DEVICE_IMPL(deform_roi_pool_forward_impl, MUSA,
                     deform_roi_pool_forward_musa);
REGISTER_DEVICE_IMPL(deform_roi_pool_backward_impl, MUSA,
                     deform_roi_pool_backward_musa);

void ActiveRotatedFilterForwardMUSAKernelLauncher(const Tensor input,
                                                  const Tensor indices,
                                                  Tensor output);

void ActiveRotatedFilterBackwardMUSAKernelLauncher(const Tensor grad_out,
                                                   const Tensor indices,
                                                   Tensor grad_in);

void active_rotated_filter_forward_musa(const Tensor input,
                                        const Tensor indices, Tensor output) {
  ActiveRotatedFilterForwardMUSAKernelLauncher(input, indices, output);
};

void active_rotated_filter_backward_musa(const Tensor grad_out,
                                         const Tensor indices, Tensor grad_in) {
  ActiveRotatedFilterBackwardMUSAKernelLauncher(grad_out, indices, grad_in);
};

void active_rotated_filter_forward_impl(const Tensor input,
                                        const Tensor indices, Tensor output);

void active_rotated_filter_backward_impl(const Tensor grad_out,
                                         const Tensor indices, Tensor grad_in);

REGISTER_DEVICE_IMPL(active_rotated_filter_forward_impl, MUSA,
                     active_rotated_filter_forward_musa);
REGISTER_DEVICE_IMPL(active_rotated_filter_backward_impl, MUSA,
                     active_rotated_filter_backward_musa);

void ConvexIoUMUSAKernelLauncher(const Tensor pointsets, const Tensor polygons,
                                 Tensor ious);

void ConvexGIoUMUSAKernelLauncher(const Tensor pointsets, const Tensor polygons,
                                  Tensor output);

void convex_iou_musa(const Tensor pointsets, const Tensor polygons,
                     Tensor ious) {
  ConvexIoUMUSAKernelLauncher(pointsets, polygons, ious);
}

void convex_giou_musa(const Tensor pointsets, const Tensor polygons,
                      Tensor output) {
  ConvexGIoUMUSAKernelLauncher(pointsets, polygons, output);
}

void convex_iou_impl(const Tensor pointsets, const Tensor polygons,
                     Tensor ious);

void convex_giou_impl(const Tensor pointsets, const Tensor polygons,
                      Tensor output);

REGISTER_DEVICE_IMPL(convex_iou_impl, MUSA, convex_iou_musa);
REGISTER_DEVICE_IMPL(convex_giou_impl, MUSA, convex_giou_musa);

Tensor DiffIoURotatedSortVerticesMUSAKernelLauncher(Tensor vertices,
                                                    Tensor mask,
                                                    Tensor num_valid);

Tensor diff_iou_rotated_sort_vertices_forward_musa(Tensor vertices, Tensor mask,
                                                   Tensor num_valid) {
  return DiffIoURotatedSortVerticesMUSAKernelLauncher(vertices, mask,
                                                      num_valid);
}

Tensor diff_iou_rotated_sort_vertices_forward_impl(Tensor vertices, Tensor mask,
                                                   Tensor num_valid);

REGISTER_DEVICE_IMPL(diff_iou_rotated_sort_vertices_forward_impl, MUSA,
                     diff_iou_rotated_sort_vertices_forward_musa);

#if ((!defined(MUSA_ARCH)) || (defined(MUSA_ARCH)) && (MUSA_ARCH > 21))
void ChamferDistanceForwardMUSAKernelLauncher(
    const Tensor xyz1, const Tensor xyz2, const Tensor dist1,
    const Tensor dist2, const Tensor idx1, const Tensor idx2);
#endif

void ChamferDistanceBackwardMUSAKernelLauncher(
    const Tensor xyz1, const Tensor xyz2, Tensor idx1, Tensor idx2,
    Tensor grad_dist1, Tensor grad_dist2, Tensor grad_xyz1, Tensor grad_xyz2);

#if ((!defined(MUSA_ARCH)) || (defined(MUSA_ARCH)) && (MUSA_ARCH > 21))
void chamfer_distance_forward_musa(const Tensor xyz1, const Tensor xyz2,
                                   const Tensor dist1, const Tensor dist2,
                                   const Tensor idx1, const Tensor idx2) {
  ChamferDistanceForwardMUSAKernelLauncher(xyz1, xyz2, dist1, dist2, idx1,
                                           idx2);
};

void chamfer_distance_backward_musa(const Tensor xyz1, const Tensor xyz2,
                                    Tensor idx1, Tensor idx2, Tensor graddist1,
                                    Tensor graddist2, Tensor gradxyz1,
                                    Tensor gradxyz2) {
  ChamferDistanceBackwardMUSAKernelLauncher(xyz1, xyz2, idx1, idx2, graddist1,
                                            graddist2, gradxyz1, gradxyz2);
};

void chamfer_distance_forward_impl(const Tensor xyz1, const Tensor xyz2,
                                   const Tensor dist1, const Tensor dist2,
                                   const Tensor idx1, const Tensor idx2);

void chamfer_distance_backward_impl(const Tensor xyz1, const Tensor xyz2,
                                    Tensor idx1, Tensor idx2, Tensor graddist1,
                                    Tensor graddist2, Tensor gradxyz1,
                                    Tensor gradxyz2);

REGISTER_DEVICE_IMPL(chamfer_distance_forward_impl, MUSA,
                     chamfer_distance_forward_musa);
REGISTER_DEVICE_IMPL(chamfer_distance_backward_impl, MUSA,
                     chamfer_distance_backward_musa);
#endif

void BezierAlignForwardMUSAKernelLauncher(Tensor input, Tensor rois,
                                          Tensor output, int aligned_height,
                                          int aligned_width,
                                          float spatial_scale,
                                          int sampling_ratio, bool aligned);

void BezierAlignBackwardMUSAKernelLauncher(
    Tensor grad_output, Tensor rois, Tensor grad_input, int aligned_height,
    int aligned_width, float spatial_scale, int sampling_ratio, bool aligned);

void bezier_align_forward_impl(Tensor input, Tensor rois, Tensor output,
                               int aligned_height, int aligned_width,
                               float spatial_scale, int sampling_ratio,
                               bool aligned);

void bezier_align_backward_impl(Tensor grad_output, Tensor rois,
                                Tensor grad_input, int aligned_height,
                                int aligned_width, float spatial_scale,
                                int sampling_ratio, bool aligned);

REGISTER_DEVICE_IMPL(bezier_align_forward_impl, MUSA,
                     BezierAlignForwardMUSAKernelLauncher);
REGISTER_DEVICE_IMPL(bezier_align_backward_impl, MUSA,
                     BezierAlignBackwardMUSAKernelLauncher);
