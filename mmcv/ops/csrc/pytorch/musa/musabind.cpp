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

void SigmoidFocalLossForwardMUSAKernelLauncher(Tensor input, Tensor target,
                                               Tensor weight, Tensor output,
                                               const float gamma,
                                               const float alpha);

void SigmoidFocalLossBackwardMUSAKernelLauncher(Tensor input, Tensor target,
                                                Tensor weight,
                                                Tensor grad_input,
                                                const float gamma,
                                                const float alpha);

void SoftmaxFocalLossForwardMUSAKernelLauncher(Tensor softmax, Tensor target,
                                               Tensor weight, Tensor output,
                                               const float gamma,
                                               const float alpha);

void SoftmaxFocalLossBackwardMUSAKernelLauncher(Tensor softmax, Tensor target,
                                                Tensor weight, Tensor buff,
                                                Tensor grad_input,
                                                const float gamma,
                                                const float alpha);

void sigmoid_focal_loss_forward_musa(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha) {
  SigmoidFocalLossForwardMUSAKernelLauncher(input, target, weight, output,
                                            gamma, alpha);
}

void sigmoid_focal_loss_backward_musa(Tensor input, Tensor target,
                                      Tensor weight, Tensor grad_input,
                                      float gamma, float alpha) {
  SigmoidFocalLossBackwardMUSAKernelLauncher(input, target, weight, grad_input,
                                             gamma, alpha);
}

void softmax_focal_loss_forward_musa(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha) {
  SoftmaxFocalLossForwardMUSAKernelLauncher(input, target, weight, output,
                                            gamma, alpha);
}

void softmax_focal_loss_backward_musa(Tensor input, Tensor target,
                                      Tensor weight, Tensor buff,
                                      Tensor grad_input, float gamma,
                                      float alpha) {
  SoftmaxFocalLossBackwardMUSAKernelLauncher(input, target, weight, buff,
                                             grad_input, gamma, alpha);
}

void sigmoid_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha);

void sigmoid_focal_loss_backward_impl(Tensor input, Tensor target,
                                      Tensor weight, Tensor grad_input,
                                      float gamma, float alpha);

void softmax_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha);

void softmax_focal_loss_backward_impl(Tensor input, Tensor target,
                                      Tensor weight, Tensor buff,
                                      Tensor grad_input, float gamma,
                                      float alpha);

REGISTER_DEVICE_IMPL(sigmoid_focal_loss_forward_impl, MUSA,
                     sigmoid_focal_loss_forward_musa);
REGISTER_DEVICE_IMPL(sigmoid_focal_loss_backward_impl, MUSA,
                     sigmoid_focal_loss_backward_musa);
REGISTER_DEVICE_IMPL(softmax_focal_loss_forward_impl, MUSA,
                     softmax_focal_loss_forward_musa);
REGISTER_DEVICE_IMPL(softmax_focal_loss_backward_impl, MUSA,
                     softmax_focal_loss_backward_musa);

void FurthestPointSamplingForwardMUSAKernelLauncher(int b, int n, int m,
                                                    const float *dataset,
                                                    float *temp, int *idxs);

void FurthestPointSamplingWithDistForwardMUSAKernelLauncher(
    int b, int n, int m, const float *dataset, float *temp, int *idxs);

void furthest_point_sampling_forward_musa(Tensor points_tensor,
                                          Tensor temp_tensor, Tensor idx_tensor,
                                          int b, int n, int m) {
  const float *dataset = points_tensor.data_ptr<float>();
  float *temp = temp_tensor.data_ptr<float>();
  int *idxs = idx_tensor.data_ptr<int>();
  FurthestPointSamplingForwardMUSAKernelLauncher(b, n, m, dataset, temp, idxs);
}

void furthest_point_sampling_with_dist_forward_musa(Tensor points_tensor,
                                                    Tensor temp_tensor,
                                                    Tensor idx_tensor, int b,
                                                    int n, int m) {
  const float *dataset = points_tensor.data_ptr<float>();
  float *temp = temp_tensor.data_ptr<float>();
  int *idxs = idx_tensor.data_ptr<int>();
  FurthestPointSamplingWithDistForwardMUSAKernelLauncher(b, n, m, dataset, temp,
                                                         idxs);
}

void furthest_point_sampling_forward_impl(Tensor points_tensor,
                                          Tensor temp_tensor, Tensor idx_tensor,
                                          int b, int n, int m);

void furthest_point_sampling_with_dist_forward_impl(Tensor points_tensor,
                                                    Tensor temp_tensor,
                                                    Tensor idx_tensor, int b,
                                                    int n, int m);

REGISTER_DEVICE_IMPL(furthest_point_sampling_forward_impl, MUSA,
                     furthest_point_sampling_forward_musa);
REGISTER_DEVICE_IMPL(furthest_point_sampling_with_dist_forward_impl, MUSA,
                     furthest_point_sampling_with_dist_forward_musa);

torch::Tensor fused_bias_leakyrelu_op(const torch::Tensor &input,
                                      const torch::Tensor &bias,
                                      const torch::Tensor &refer, int act,
                                      int grad, float alpha, float scale);

torch::Tensor fused_bias_leakyrelu_op_impl(const torch::Tensor &input,
                                           const torch::Tensor &bias,
                                           const torch::Tensor &refer, int act,
                                           int grad, float alpha, float scale);
REGISTER_DEVICE_IMPL(fused_bias_leakyrelu_op_impl, MUSA,
                     fused_bias_leakyrelu_op);

torch::Tensor bias_act_op_impl(const torch::Tensor &input,
                               const torch::Tensor &bias,
                               const torch::Tensor &xref,
                               const torch::Tensor &yref,
                               const torch::Tensor &dy, int grad, int dim,
                               int act, float alpha, float gain, float clamp);

torch::Tensor bias_act_op(const torch::Tensor &input, const torch::Tensor &bias,
                          const torch::Tensor &xref, const torch::Tensor &yref,
                          const torch::Tensor &dy, int grad, int dim, int act,
                          float alpha, float gain, float clamp);

REGISTER_DEVICE_IMPL(bias_act_op_impl, MUSA, bias_act_op);

void GatherPointsForwardMUSAKernelLauncher(int b, int c, int n, int npoints,
                                           const Tensor points,
                                           const Tensor idx, Tensor out);

void GatherPointsBackwardMUSAKernelLauncher(int b, int c, int n, int npoints,
                                            const Tensor grad_out,
                                            const Tensor idx,
                                            Tensor grad_points);

void gather_points_forward_musa(int b, int c, int n, int npoints,
                                const Tensor points, const Tensor idx,
                                Tensor out) {
  GatherPointsForwardMUSAKernelLauncher(b, c, n, npoints, points, idx, out);
};

void gather_points_backward_musa(int b, int c, int n, int npoints,
                                 const Tensor grad_out, const Tensor idx,
                                 Tensor grad_points) {
  GatherPointsBackwardMUSAKernelLauncher(b, c, n, npoints, grad_out, idx,
                                         grad_points);
};

void gather_points_forward_impl(int b, int c, int n, int npoints,
                                const Tensor points, const Tensor idx,
                                Tensor out);

void gather_points_backward_impl(int b, int c, int n, int npoints,
                                 const Tensor grad_out, const Tensor idx,
                                 Tensor grad_points);

REGISTER_DEVICE_IMPL(gather_points_forward_impl, MUSA,
                     gather_points_forward_musa);
REGISTER_DEVICE_IMPL(gather_points_backward_impl, MUSA,
                     gather_points_backward_musa);

void GroupPointsForwardMUSAKernelLauncher(int b, int c, int n, int npoints,
                                          int nsample, const Tensor points,
                                          const Tensor idx, Tensor out);

void GroupPointsBackwardMUSAKernelLauncher(int b, int c, int n, int npoints,
                                           int nsample, const Tensor grad_out,
                                           const Tensor idx,
                                           Tensor grad_points);

void group_points_forward_musa(int b, int c, int n, int npoints, int nsample,
                               const Tensor points, const Tensor idx,
                               Tensor out) {
  GroupPointsForwardMUSAKernelLauncher(b, c, n, npoints, nsample, points, idx,
                                       out);
};

void group_points_backward_musa(int b, int c, int n, int npoints, int nsample,
                                const Tensor grad_out, const Tensor idx,
                                Tensor grad_points) {
  GroupPointsBackwardMUSAKernelLauncher(b, c, n, npoints, nsample, grad_out,
                                        idx, grad_points);
};

void group_points_forward_impl(int b, int c, int n, int npoints, int nsample,
                               const Tensor points, const Tensor idx,
                               Tensor out);

void group_points_backward_impl(int b, int c, int n, int npoints, int nsample,
                                const Tensor grad_out, const Tensor idx,
                                Tensor grad_points);

REGISTER_DEVICE_IMPL(group_points_forward_impl, MUSA,
                     group_points_forward_musa);
REGISTER_DEVICE_IMPL(group_points_backward_impl, MUSA,
                     group_points_backward_musa);

void StackGroupPointsForwardMUSAKernelLauncher(
    int b, int c, int m, int nsample, const Tensor features_tensor,
    const Tensor features_batch_cnt_tensor, const Tensor idx_tensor,
    const Tensor idx_batch_cnt_tensor, Tensor out_tensor);
void StackGroupPointsBackwardMUSAKernelLauncher(
    int b, int c, int m, int n, int nsample, const Tensor grad_out_tensor,
    const Tensor idx_tensor, const Tensor idx_batch_cnt_tensor,
    const Tensor features_batch_cnt_tensor, Tensor grad_features_tensor);

void stack_group_points_forward_musa(int b, int c, int m, int nsample,
                                     const Tensor features_tensor,
                                     const Tensor features_batch_cnt_tensor,
                                     const Tensor idx_tensor,
                                     const Tensor idx_batch_cnt_tensor,
                                     Tensor out_tensor) {
  StackGroupPointsForwardMUSAKernelLauncher(
      b, c, m, nsample, features_tensor, features_batch_cnt_tensor, idx_tensor,
      idx_batch_cnt_tensor, out_tensor);
};

void stack_group_points_backward_musa(int b, int c, int m, int n, int nsample,
                                      const Tensor grad_out_tensor,
                                      const Tensor idx_tensor,
                                      const Tensor idx_batch_cnt_tensor,
                                      const Tensor features_batch_cnt_tensor,
                                      Tensor grad_features_tensor) {
  StackGroupPointsBackwardMUSAKernelLauncher(
      b, c, m, n, nsample, grad_out_tensor, idx_tensor, idx_batch_cnt_tensor,
      features_batch_cnt_tensor, grad_features_tensor);
};

void stack_group_points_forward_impl(int b, int c, int m, int nsample,
                                     const Tensor features_tensor,
                                     const Tensor features_batch_cnt_tensor,
                                     const Tensor idx_tensor,
                                     const Tensor idx_batch_cnt_tensor,
                                     Tensor out_tensor);

void stack_group_points_backward_impl(int b, int c, int m, int n, int nsample,
                                      const Tensor grad_out_tensor,
                                      const Tensor idx_tensor,
                                      const Tensor idx_batch_cnt_tensor,
                                      const Tensor features_batch_cnt_tensor,
                                      Tensor grad_features_tensor);

REGISTER_DEVICE_IMPL(stack_group_points_forward_impl, MUSA,
                     stack_group_points_forward_musa);
REGISTER_DEVICE_IMPL(stack_group_points_backward_impl, MUSA,
                     stack_group_points_backward_musa);

void IoU3DBoxesOverlapBevForwardMUSAKernelLauncher(const int num_a,
                                                   const Tensor boxes_a,
                                                   const int num_b,
                                                   const Tensor boxes_b,
                                                   Tensor ans_overlap);

void IoU3DNMS3DForwardMUSAKernelLauncher(const Tensor boxes, Tensor &keep,
                                         Tensor &keep_num,
                                         float nms_overlap_thresh);

void IoU3DNMS3DNormalForwardMUSAKernelLauncher(const Tensor boxes, Tensor &keep,
                                               Tensor &keep_num,
                                               float nms_overlap_thresh);

void iou3d_boxes_overlap_bev_forward_musa(const int num_a, const Tensor boxes_a,
                                          const int num_b, const Tensor boxes_b,
                                          Tensor ans_overlap) {
  IoU3DBoxesOverlapBevForwardMUSAKernelLauncher(num_a, boxes_a, num_b, boxes_b,
                                                ans_overlap);
};

void iou3d_nms3d_forward_musa(const Tensor boxes, Tensor &keep,
                              Tensor &keep_num, float nms_overlap_thresh) {
  IoU3DNMS3DForwardMUSAKernelLauncher(boxes, keep, keep_num,
                                      nms_overlap_thresh);
};

void iou3d_nms3d_normal_forward_musa(const Tensor boxes, Tensor &keep,
                                     Tensor &keep_num,
                                     float nms_overlap_thresh) {
  IoU3DNMS3DNormalForwardMUSAKernelLauncher(boxes, keep, keep_num,
                                            nms_overlap_thresh);
};

void iou3d_boxes_overlap_bev_forward_impl(const int num_a, const Tensor boxes_a,
                                          const int num_b, const Tensor boxes_b,
                                          Tensor ans_overlap);

void iou3d_nms3d_forward_impl(const Tensor boxes, Tensor &keep,
                              Tensor &keep_num, float nms_overlap_thresh);

void iou3d_nms3d_normal_forward_impl(const Tensor boxes, Tensor &keep,
                                     Tensor &keep_num,
                                     float nms_overlap_thresh);

REGISTER_DEVICE_IMPL(iou3d_boxes_overlap_bev_forward_impl, MUSA,
                     iou3d_boxes_overlap_bev_forward_musa);
REGISTER_DEVICE_IMPL(iou3d_nms3d_forward_impl, MUSA, iou3d_nms3d_forward_musa);
REGISTER_DEVICE_IMPL(iou3d_nms3d_normal_forward_impl, MUSA,
                     iou3d_nms3d_normal_forward_musa);

void KNNForwardMUSAKernelLauncher(int b, int n, int m, int nsample,
                                  const Tensor xyz, const Tensor new_xyz,
                                  Tensor idx, Tensor dist2);

void knn_forward_musa(int b, int n, int m, int nsample, const Tensor xyz,
                      const Tensor new_xyz, Tensor idx, Tensor dist2) {
  KNNForwardMUSAKernelLauncher(b, n, m, nsample, xyz, new_xyz, idx, dist2);
}

void knn_forward_impl(int b, int n, int m, int nsample, const Tensor xyz,
                      const Tensor new_xyz, Tensor idx, Tensor dist2);
REGISTER_DEVICE_IMPL(knn_forward_impl, MUSA, knn_forward_musa);

void MaskedIm2colForwardMUSAKernelLauncher(const Tensor bottom_data,
                                           const Tensor mask_h_idx,
                                           const Tensor mask_w_idx,
                                           Tensor top_data, const int kernel_h,
                                           const int kernel_w, const int pad_h,
                                           const int pad_w);

void MaskedCol2imForwardMUSAKernelLauncher(const Tensor bottom_data,
                                           const Tensor mask_h_idx,
                                           const Tensor mask_w_idx,
                                           Tensor top_data, const int height,
                                           const int width, const int channels);

void masked_im2col_forward_musa(const Tensor im, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor col,
                                const int kernel_h, const int kernel_w,
                                const int pad_h, const int pad_w) {
  // im: (n, ic, h, w), kernel size (kh, kw)
  // kernel: (oc, ic * kh * kw), col: (kh * kw * ic, ow * oh)
  MaskedIm2colForwardMUSAKernelLauncher(im, mask_h_idx, mask_w_idx, col,
                                        kernel_h, kernel_w, pad_h, pad_w);
}

void masked_col2im_forward_musa(const Tensor col, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor im, int height,
                                int width, int channels) {
  // im: (n, ic, h, w), kernel size (kh, kw)
  // kernel: (oc, ic * kh * kh), col: (kh * kw * ic, ow * oh)
  MaskedCol2imForwardMUSAKernelLauncher(col, mask_h_idx, mask_w_idx, im, height,
                                        width, channels);
}

void masked_im2col_forward_impl(const Tensor im, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor col,
                                const int kernel_h, const int kernel_w,
                                const int pad_h, const int pad_w);

void masked_col2im_forward_impl(const Tensor col, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor im, int height,
                                int width, int channels);

REGISTER_DEVICE_IMPL(masked_im2col_forward_impl, MUSA,
                     masked_im2col_forward_musa);
REGISTER_DEVICE_IMPL(masked_col2im_forward_impl, MUSA,
                     masked_col2im_forward_musa);

void modulated_deformable_im2col_musa(
    const Tensor data_im, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col);

void modulated_deformable_col2im_musa(
    const Tensor data_col, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im);

void modulated_deformable_col2im_coord_musa(
    const Tensor data_col, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    Tensor grad_offset, Tensor grad_mask);

void modulated_deformable_im2col_impl(
    const Tensor data_im, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col);

void modulated_deformable_col2im_impl(
    const Tensor data_col, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im);

void modulated_deformable_col2im_coord_impl(
    const Tensor data_col, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    Tensor grad_offset, Tensor grad_mask);

REGISTER_DEVICE_IMPL(modulated_deformable_im2col_impl, MUSA,
                     modulated_deformable_im2col_musa);
REGISTER_DEVICE_IMPL(modulated_deformable_col2im_impl, MUSA,
                     modulated_deformable_col2im_musa);
REGISTER_DEVICE_IMPL(modulated_deformable_col2im_coord_impl, MUSA,
                     modulated_deformable_col2im_coord_musa);

Tensor ms_deform_attn_musa_forward(const Tensor &value,
                                   const Tensor &spatial_shapes,
                                   const Tensor &level_start_index,
                                   const Tensor &sampling_loc,
                                   const Tensor &attn_weight,
                                   const int im2col_step);

void ms_deform_attn_musa_backward(
    const Tensor &value, const Tensor &spatial_shapes,
    const Tensor &level_start_index, const Tensor &sampling_loc,
    const Tensor &attn_weight, const Tensor &grad_output, Tensor &grad_value,
    Tensor &grad_sampling_loc, Tensor &grad_attn_weight, const int im2col_step);

Tensor ms_deform_attn_impl_forward(const Tensor &value,
                                   const Tensor &spatial_shapes,
                                   const Tensor &level_start_index,
                                   const Tensor &sampling_loc,
                                   const Tensor &attn_weight,
                                   const int im2col_step);

void ms_deform_attn_impl_backward(
    const Tensor &value, const Tensor &spatial_shapes,
    const Tensor &level_start_index, const Tensor &sampling_loc,
    const Tensor &attn_weight, const Tensor &grad_output, Tensor &grad_value,
    Tensor &grad_sampling_loc, Tensor &grad_attn_weight, const int im2col_step);

REGISTER_DEVICE_IMPL(ms_deform_attn_impl_forward, MUSA,
                     ms_deform_attn_musa_forward);
REGISTER_DEVICE_IMPL(ms_deform_attn_impl_backward, MUSA,
                     ms_deform_attn_musa_backward);

Tensor NMSMUSAKernelLauncher(Tensor boxes, Tensor scores, float iou_threshold,
                             int offset);

Tensor nms_musa(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
  return NMSMUSAKernelLauncher(boxes, scores, iou_threshold, offset);
}

Tensor nms_impl(Tensor boxes, Tensor scores, float iou_threshold, int offset);
REGISTER_DEVICE_IMPL(nms_impl, MUSA, nms_musa);

void MinAreaPolygonsMUSAKernelLauncher(const Tensor pointsets, Tensor polygons);

void min_area_polygons_musa(const Tensor pointsets, Tensor polygons) {
  MinAreaPolygonsMUSAKernelLauncher(pointsets, polygons);
}

void min_area_polygons_impl(const Tensor pointsets, Tensor polygons);

REGISTER_DEVICE_IMPL(min_area_polygons_impl, MUSA, min_area_polygons_musa);

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
