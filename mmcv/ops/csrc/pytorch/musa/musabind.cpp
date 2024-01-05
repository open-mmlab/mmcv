#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#include <iostream>
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
  std::cout<<"ball_query_forward_musa"<<std::endl;
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

torch::Tensor filtered_lrelu_act_op_impl(torch::Tensor x, torch::Tensor si,
                                         int sx, int sy, float gain,
                                         float slope, float clamp,
                                         bool writeSigns);

torch::Tensor filtered_lrelu_act_op(torch::Tensor x, torch::Tensor si, int sx,
                                    int sy, float gain, float slope,
                                    float clamp, bool writeSigns);

REGISTER_DEVICE_IMPL(filtered_lrelu_act_op_impl, MUSA, filtered_lrelu_act_op);

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

void PointsInBoxesPartForwardMUSAKernelLauncher(int batch_size, int boxes_num,
                                                int pts_num, const Tensor boxes,
                                                const Tensor pts,
                                                Tensor box_idx_of_points);

void PointsInBoxesAllForwardMUSAKernelLauncher(int batch_size, int boxes_num,
                                               int pts_num, const Tensor boxes,
                                               const Tensor pts,
                                               Tensor box_idx_of_points);

void points_in_boxes_part_forward_musa(int batch_size, int boxes_num,
                                       int pts_num, const Tensor boxes,
                                       const Tensor pts,
                                       Tensor box_idx_of_points) {
  PointsInBoxesPartForwardMUSAKernelLauncher(batch_size, boxes_num, pts_num,
                                             boxes, pts, box_idx_of_points);
};

void points_in_boxes_all_forward_musa(int batch_size, int boxes_num,
                                      int pts_num, const Tensor boxes,
                                      const Tensor pts,
                                      Tensor box_idx_of_points) {
  PointsInBoxesAllForwardMUSAKernelLauncher(batch_size, boxes_num, pts_num,
                                            boxes, pts, box_idx_of_points);
};

void points_in_boxes_part_forward_impl(int batch_size, int boxes_num,
                                       int pts_num, const Tensor boxes,
                                       const Tensor pts,
                                       Tensor box_idx_of_points);

void points_in_boxes_all_forward_impl(int batch_size, int boxes_num,
                                      int pts_num, const Tensor boxes,
                                      const Tensor pts,
                                      Tensor box_idx_of_points);
REGISTER_DEVICE_IMPL(points_in_boxes_part_forward_impl, MUSA,
                     points_in_boxes_part_forward_musa);
REGISTER_DEVICE_IMPL(points_in_boxes_all_forward_impl, MUSA,
                     points_in_boxes_all_forward_musa);

void PSAMaskForwardMUSAKernelLauncher(const int psa_type, const Tensor input,
                                      Tensor output, const int num_,
                                      const int h_feature, const int w_feature,
                                      const int h_mask, const int w_mask,
                                      const int half_h_mask,
                                      const int half_w_mask);

void PSAMaskBackwardMUSAKernelLauncher(
    const int psa_type, const Tensor grad_output, Tensor grad_input,
    const int num_, const int h_feature, const int w_feature, const int h_mask,
    const int w_mask, const int half_h_mask, const int half_w_mask);

void psamask_forward_musa(const int psa_type, const Tensor input, Tensor output,
                          const int num_, const int h_feature,
                          const int w_feature, const int h_mask,
                          const int w_mask, const int half_h_mask,
                          const int half_w_mask) {
  PSAMaskForwardMUSAKernelLauncher(psa_type, input, output, num_, h_feature,
                                   w_feature, h_mask, w_mask, half_h_mask,
                                   half_w_mask);
}

void psamask_backward_musa(const int psa_type, const Tensor grad_output,
                           Tensor grad_input, const int num_,
                           const int h_feature, const int w_feature,
                           const int h_mask, const int w_mask,
                           const int half_h_mask, const int half_w_mask) {
  PSAMaskBackwardMUSAKernelLauncher(psa_type, grad_output, grad_input, num_,
                                    h_feature, w_feature, h_mask, w_mask,
                                    half_h_mask, half_w_mask);
}

void psamask_forward_impl(const int psa_type, const Tensor input, Tensor output,
                          const int num_, const int h_feature,
                          const int w_feature, const int h_mask,
                          const int w_mask, const int half_h_mask,
                          const int half_w_mask);

void psamask_backward_impl(const int psa_type, const Tensor grad_output,
                           Tensor grad_input, const int num_,
                           const int h_feature, const int w_feature,
                           const int h_mask, const int w_mask,
                           const int half_h_mask, const int half_w_mask);
REGISTER_DEVICE_IMPL(psamask_forward_impl, MUSA, psamask_forward_musa);
REGISTER_DEVICE_IMPL(psamask_backward_impl, MUSA, psamask_backward_musa);

void ROIAlignForwardMUSAKernelLauncher(Tensor input, Tensor rois, Tensor output,
                                       Tensor argmax_y, Tensor argmax_x,
                                       int aligned_height, int aligned_width,
                                       float spatial_scale, int sampling_ratio,
                                       int pool_mode, bool aligned);

void ROIAlignBackwardMUSAKernelLauncher(Tensor grad_output, Tensor rois,
                                        Tensor argmax_y, Tensor argmax_x,
                                        Tensor grad_input, int aligned_height,
                                        int aligned_width, float spatial_scale,
                                        int sampling_ratio, int pool_mode,
                                        bool aligned);

void roi_align_forward_musa(Tensor input, Tensor rois, Tensor output,
                            Tensor argmax_y, Tensor argmax_x,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned) {
  ROIAlignForwardMUSAKernelLauncher(
      input, rois, output, argmax_y, argmax_x, aligned_height, aligned_width,
      spatial_scale, sampling_ratio, pool_mode, aligned);
}

void roi_align_backward_musa(Tensor grad_output, Tensor rois, Tensor argmax_y,
                             Tensor argmax_x, Tensor grad_input,
                             int aligned_height, int aligned_width,
                             float spatial_scale, int sampling_ratio,
                             int pool_mode, bool aligned) {
  ROIAlignBackwardMUSAKernelLauncher(
      grad_output, rois, argmax_y, argmax_x, grad_input, aligned_height,
      aligned_width, spatial_scale, sampling_ratio, pool_mode, aligned);
}

void roi_align_forward_impl(Tensor input, Tensor rois, Tensor output,
                            Tensor argmax_y, Tensor argmax_x,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned);

void roi_align_backward_impl(Tensor grad_output, Tensor rois, Tensor argmax_y,
                             Tensor argmax_x, Tensor grad_input,
                             int aligned_height, int aligned_width,
                             float spatial_scale, int sampling_ratio,
                             int pool_mode, bool aligned);

REGISTER_DEVICE_IMPL(roi_align_forward_impl, MUSA, roi_align_forward_musa);
REGISTER_DEVICE_IMPL(roi_align_backward_impl, MUSA, roi_align_backward_musa);

void ROIAlignRotatedForwardMUSAKernelLauncher(
    const at::Tensor input, const at::Tensor rois, const float spatial_scale,
    const int sampling_ratio, const bool aligned, const bool clockwise,
    const int channels, const int height, const int width, const int num_rois,
    const int pooled_height, const int pooled_width, at::Tensor output);

void ROIAlignRotatedBackwardMUSAKernelLauncher(
    const at::Tensor top_grad, const at::Tensor rois, const float spatial_scale,
    const int sampling_ratio, const bool aligned, const bool clockwise,
    const int channels, const int height, const int width, const int num_rois,
    const int pooled_height, const int pooled_width, at::Tensor bottom_grad);

void roi_align_rotated_forward_musa(Tensor input, Tensor rois, Tensor output,
                                    int aligned_height, int aligned_width,
                                    float spatial_scale, int sampling_ratio,
                                    bool aligned, bool clockwise) {
  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);

  if (size_rois != 6) {
    AT_ERROR("wrong roi size");
  }

  int num_channels = input.size(1);
  int data_height = input.size(2);
  int data_width = input.size(3);
  ROIAlignRotatedForwardMUSAKernelLauncher(
      input, rois, spatial_scale, sampling_ratio, aligned, clockwise,
      num_channels, data_height, data_width, num_rois, aligned_height,
      aligned_width, output);
}

void roi_align_rotated_backward_musa(Tensor top_grad, Tensor rois,
                                     Tensor bottom_grad, int aligned_height,
                                     int aligned_width, float spatial_scale,
                                     int sampling_ratio, bool aligned,
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
  ROIAlignRotatedBackwardMUSAKernelLauncher(
      top_grad, rois, spatial_scale, sampling_ratio, aligned, clockwise,
      num_channels, data_height, data_width, num_rois, aligned_height,
      aligned_width, bottom_grad);
}

void roi_align_rotated_forward_impl(Tensor input, Tensor rois, Tensor output,
                                    int aligned_height, int aligned_width,
                                    float spatial_scale, int sampling_ratio,
                                    bool aligned, bool clockwise);

void roi_align_rotated_backward_impl(Tensor top_grad, Tensor rois,
                                     Tensor bottom_grad, int aligned_height,
                                     int aligned_width, float spatial_scale,
                                     int sampling_ratio, bool aligned,
                                     bool clockwise);
REGISTER_DEVICE_IMPL(roi_align_rotated_forward_impl, MUSA,
                     roi_align_rotated_forward_musa);
REGISTER_DEVICE_IMPL(roi_align_rotated_backward_impl, MUSA,
                     roi_align_rotated_backward_musa);

void RiROIAlignRotatedForwardMUSAKernelLauncher(
    const at::Tensor features, const at::Tensor rois, const float spatial_scale,
    const int num_samples, const bool clockwise, const int channels,
    const int height, const int width, const int num_rois,
    const int pooled_height, const int pooled_width, const int num_orientations,
    at::Tensor output);

void RiROIAlignRotatedBackwardMUSAKernelLauncher(
    const at::Tensor top_grad, const at::Tensor rois, const float spatial_scale,
    const int num_samples, const bool clockwise, const int channels,
    const int height, const int width, const int num_rois,
    const int pooled_height, const int pooled_width, const int num_orientations,
    at::Tensor bottom_grad);

void riroi_align_rotated_forward_musa(Tensor features, Tensor rois,
                                      Tensor output, int pooled_height,
                                      int pooled_width, float spatial_scale,
                                      int num_samples, int num_orientations,
                                      bool clockwise) {
  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);
  if (size_rois != 6) {
    AT_ERROR("wrong roi size");
  }
  CHECK_CONTIGUOUS(features);
  CHECK_CONTIGUOUS(rois);
  int num_channels = features.size(1) / num_orientations;
  int data_height = features.size(2);
  int data_width = features.size(3);
  RiROIAlignRotatedForwardMUSAKernelLauncher(
      features, rois, spatial_scale, num_samples, clockwise, num_channels,
      data_height, data_width, num_rois, pooled_height, pooled_width,
      num_orientations, output);
}

void riroi_align_rotated_backward_musa(Tensor top_grad, Tensor rois,
                                       Tensor bottom_grad, int pooled_height,
                                       int pooled_width, float spatial_scale,
                                       int num_samples, int num_orientations,
                                       bool clockwise) {
  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);
  if (size_rois != 6) {
    AT_ERROR("wrong roi size");
  }
  CHECK_CONTIGUOUS(top_grad);
  CHECK_CONTIGUOUS(rois);
  int num_channels = bottom_grad.size(1) / num_orientations;
  int data_height = bottom_grad.size(2);
  int data_width = bottom_grad.size(3);
  RiROIAlignRotatedBackwardMUSAKernelLauncher(
      top_grad, rois, spatial_scale, num_samples, clockwise, num_channels,
      data_height, data_width, num_rois, pooled_height, pooled_width,
      num_orientations, bottom_grad);
}

void riroi_align_rotated_forward_impl(Tensor features, Tensor rois,
                                      Tensor output, int pooled_height,
                                      int pooled_width, float spatial_scale,
                                      int num_samples, int num_orientations,
                                      bool clockwise);

void riroi_align_rotated_backward_impl(Tensor top_grad, Tensor rois,
                                       Tensor bottom_grad, int pooled_height,
                                       int pooled_width, float spatial_scale,
                                       int num_samples, int num_orientations,
                                       bool clockwise);

REGISTER_DEVICE_IMPL(riroi_align_rotated_forward_impl, MUSA,
                     riroi_align_rotated_forward_musa);
REGISTER_DEVICE_IMPL(riroi_align_rotated_backward_impl, MUSA,
                     riroi_align_rotated_backward_musa);

void RoiawarePool3dForwardMUSAKernelLauncher(
    int boxes_num, int pts_num, int channels, int max_pts_each_voxel, int out_x,
    int out_y, int out_z, const Tensor rois, const Tensor pts,
    const Tensor pts_feature, Tensor argmax, Tensor pts_idx_of_voxels,
    Tensor pooled_features, int pool_method);

void RoiawarePool3dBackwardMUSAKernelLauncher(
    int boxes_num, int out_x, int out_y, int out_z, int channels,
    int max_pts_each_voxel, const Tensor pts_idx_of_voxels, const Tensor argmax,
    const Tensor grad_out, Tensor grad_in, int pool_method);

void roiaware_pool3d_forward_musa(int boxes_num, int pts_num, int channels,
                                  int max_pts_each_voxel, int out_x, int out_y,
                                  int out_z, const Tensor rois,
                                  const Tensor pts, const Tensor pts_feature,
                                  Tensor argmax, Tensor pts_idx_of_voxels,
                                  Tensor pooled_features, int pool_method) {
  RoiawarePool3dForwardMUSAKernelLauncher(
      boxes_num, pts_num, channels, max_pts_each_voxel, out_x, out_y, out_z,
      rois, pts, pts_feature, argmax, pts_idx_of_voxels, pooled_features,
      pool_method);
};

void roiaware_pool3d_backward_musa(int boxes_num, int out_x, int out_y,
                                   int out_z, int channels,
                                   int max_pts_each_voxel,
                                   const Tensor pts_idx_of_voxels,
                                   const Tensor argmax, const Tensor grad_out,
                                   Tensor grad_in, int pool_method) {
  RoiawarePool3dBackwardMUSAKernelLauncher(
      boxes_num, out_x, out_y, out_z, channels, max_pts_each_voxel,
      pts_idx_of_voxels, argmax, grad_out, grad_in, pool_method);
};

void roiaware_pool3d_forward_impl(int boxes_num, int pts_num, int channels,
                                  int max_pts_each_voxel, int out_x, int out_y,
                                  int out_z, const Tensor rois,
                                  const Tensor pts, const Tensor pts_feature,
                                  Tensor argmax, Tensor pts_idx_of_voxels,
                                  Tensor pooled_features, int pool_method);

void roiaware_pool3d_backward_impl(int boxes_num, int out_x, int out_y,
                                   int out_z, int channels,
                                   int max_pts_each_voxel,
                                   const Tensor pts_idx_of_voxels,
                                   const Tensor argmax, const Tensor grad_out,
                                   Tensor grad_in, int pool_method);

REGISTER_DEVICE_IMPL(roiaware_pool3d_forward_impl, MUSA,
                     roiaware_pool3d_forward_musa);
REGISTER_DEVICE_IMPL(roiaware_pool3d_backward_impl, MUSA,
                     roiaware_pool3d_backward_musa);

void RoIPointPool3dForwardMUSAKernelLauncher(
    int batch_size, int pts_num, int boxes_num, int feature_in_len,
    int sampled_pts_num, const Tensor xyz, const Tensor boxes3d,
    const Tensor pts_feature, Tensor pooled_features, Tensor pooled_empty_flag);

void roipoint_pool3d_forward_musa(int batch_size, int pts_num, int boxes_num,
                                  int feature_in_len, int sampled_pts_num,
                                  const Tensor xyz, const Tensor boxes3d,
                                  const Tensor pts_feature,
                                  Tensor pooled_features,
                                  Tensor pooled_empty_flag) {
  RoIPointPool3dForwardMUSAKernelLauncher(
      batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num, xyz,
      boxes3d, pts_feature, pooled_features, pooled_empty_flag);
};

void roipoint_pool3d_forward_impl(int batch_size, int pts_num, int boxes_num,
                                  int feature_in_len, int sampled_pts_num,
                                  const Tensor xyz, const Tensor boxes3d,
                                  const Tensor pts_feature,
                                  Tensor pooled_features,
                                  Tensor pooled_empty_flag);
REGISTER_DEVICE_IMPL(roipoint_pool3d_forward_impl, MUSA,
                     roipoint_pool3d_forward_musa);

void ROIPoolForwardMUSAKernelLauncher(Tensor input, Tensor rois, Tensor output,
                                      Tensor argmax, int pooled_height,
                                      int pooled_width, float spatial_scale);

void ROIPoolBackwardMUSAKernelLauncher(Tensor grad_output, Tensor rois,
                                       Tensor argmax, Tensor grad_input,
                                       int pooled_height, int pooled_width,
                                       float spatial_scale);

void roi_pool_forward_musa(Tensor input, Tensor rois, Tensor output,
                           Tensor argmax, int pooled_height, int pooled_width,
                           float spatial_scale) {
  ROIPoolForwardMUSAKernelLauncher(input, rois, output, argmax, pooled_height,
                                   pooled_width, spatial_scale);
}

void roi_pool_backward_musa(Tensor grad_output, Tensor rois, Tensor argmax,
                            Tensor grad_input, int pooled_height,
                            int pooled_width, float spatial_scale) {
  ROIPoolBackwardMUSAKernelLauncher(grad_output, rois, argmax, grad_input,
                                    pooled_height, pooled_width, spatial_scale);
}

void roi_pool_forward_impl(Tensor input, Tensor rois, Tensor output,
                           Tensor argmax, int pooled_height, int pooled_width,
                           float spatial_scale);
void roi_pool_backward_impl(Tensor grad_output, Tensor rois, Tensor argmax,
                            Tensor grad_input, int pooled_height,
                            int pooled_width, float spatial_scale);
REGISTER_DEVICE_IMPL(roi_pool_forward_impl, MUSA, roi_pool_forward_musa);
REGISTER_DEVICE_IMPL(roi_pool_backward_impl, MUSA, roi_pool_backward_musa);

typedef enum { SUM = 0, MEAN = 1, MAX = 2 } reduce_t;

std::vector<at::Tensor> DynamicPointToVoxelForwardMUSAKernelLauncher(
    const at::Tensor &feats, const at::Tensor &coors,
    const reduce_t reduce_type);

void DynamicPointToVoxelBackwardMUSAKernelLauncher(
    at::Tensor &grad_feats, const at::Tensor &grad_reduced_feats,
    const at::Tensor &feats, const at::Tensor &reduced_feats,
    const at::Tensor &coors_map, const at::Tensor &reduce_count,
    const reduce_t reduce_type);

std::vector<torch::Tensor> dynamic_point_to_voxel_forward_musa(
    const torch::Tensor &feats, const torch::Tensor &coors,
    const reduce_t reduce_type) {
  return DynamicPointToVoxelForwardMUSAKernelLauncher(feats, coors,
                                                      reduce_type);
};

void dynamic_point_to_voxel_backward_musa(
    torch::Tensor &grad_feats, const torch::Tensor &grad_reduced_feats,
    const torch::Tensor &feats, const torch::Tensor &reduced_feats,
    const torch::Tensor &coors_idx, const torch::Tensor &reduce_count,
    const reduce_t reduce_type) {
  DynamicPointToVoxelBackwardMUSAKernelLauncher(grad_feats, grad_reduced_feats,
                                                feats, reduced_feats, coors_idx,
                                                reduce_count, reduce_type);
};

std::vector<torch::Tensor> dynamic_point_to_voxel_forward_impl(
    const torch::Tensor &feats, const torch::Tensor &coors,
    const reduce_t reduce_type);

void dynamic_point_to_voxel_backward_impl(
    torch::Tensor &grad_feats, const torch::Tensor &grad_reduced_feats,
    const torch::Tensor &feats, const torch::Tensor &reduced_feats,
    const torch::Tensor &coors_idx, const torch::Tensor &reduce_count,
    const reduce_t reduce_type);

REGISTER_DEVICE_IMPL(dynamic_point_to_voxel_forward_impl, MUSA,
                     dynamic_point_to_voxel_forward_musa);
REGISTER_DEVICE_IMPL(dynamic_point_to_voxel_backward_impl, MUSA,
                     dynamic_point_to_voxel_backward_musa);

void SyncBNForwardMeanMUSAKernelLauncher(const Tensor input, Tensor mean);

void SyncBNForwardVarMUSAKernelLauncher(const Tensor input, const Tensor mean,
                                        Tensor var);

void SyncBNForwardOutputMUSAKernelLauncher(
    const Tensor input, const Tensor mean, const Tensor var,
    Tensor running_mean, Tensor running_var, const Tensor weight,
    const Tensor bias, Tensor norm, Tensor std, Tensor output, float eps,
    float momentum, int group_size);

void SyncBNBackwardParamMUSAKernelLauncher(const Tensor grad_output,
                                           const Tensor norm,
                                           Tensor grad_weight,
                                           Tensor grad_bias);

void SyncBNBackwardDataMUSAKernelLauncher(const Tensor grad_output,
                                          const Tensor weight,
                                          const Tensor grad_weight,
                                          const Tensor grad_bias,
                                          const Tensor norm, const Tensor std,
                                          Tensor grad_input);

void sync_bn_forward_mean_musa(const Tensor input, Tensor mean) {
  SyncBNForwardMeanMUSAKernelLauncher(input, mean);
}

void sync_bn_forward_var_musa(const Tensor input, const Tensor mean,
                              Tensor var) {
  SyncBNForwardVarMUSAKernelLauncher(input, mean, var);
}

void sync_bn_forward_output_musa(const Tensor input, const Tensor mean,
                                 const Tensor var, Tensor running_mean,
                                 Tensor running_var, const Tensor weight,
                                 const Tensor bias, Tensor norm, Tensor std,
                                 Tensor output, float eps, float momentum,
                                 int group_size) {
  SyncBNForwardOutputMUSAKernelLauncher(input, mean, var, running_mean,
                                        running_var, weight, bias, norm, std,
                                        output, eps, momentum, group_size);
}

void sync_bn_backward_param_musa(const Tensor grad_output, const Tensor norm,
                                 Tensor grad_weight, Tensor grad_bias) {
  SyncBNBackwardParamMUSAKernelLauncher(grad_output, norm, grad_weight,
                                        grad_bias);
}

void sync_bn_backward_data_musa(const Tensor grad_output, const Tensor weight,
                                const Tensor grad_weight,
                                const Tensor grad_bias, const Tensor norm,
                                const Tensor std, Tensor grad_input) {
  SyncBNBackwardDataMUSAKernelLauncher(grad_output, weight, grad_weight,
                                       grad_bias, norm, std, grad_input);
}

void sync_bn_forward_mean_impl(const Tensor input, Tensor mean);

void sync_bn_forward_var_impl(const Tensor input, const Tensor mean,
                              Tensor var);

void sync_bn_forward_output_impl(const Tensor input, const Tensor mean,
                                 const Tensor var, Tensor running_mean,
                                 Tensor running_var, const Tensor weight,
                                 const Tensor bias, Tensor norm, Tensor std,
                                 Tensor output, float eps, float momentum,
                                 int group_size);

void sync_bn_backward_param_impl(const Tensor grad_output, const Tensor norm,
                                 Tensor grad_weight, Tensor grad_bias);

void sync_bn_backward_data_impl(const Tensor grad_output, const Tensor weight,
                                const Tensor grad_weight,
                                const Tensor grad_bias, const Tensor norm,
                                const Tensor std, Tensor grad_input);

REGISTER_DEVICE_IMPL(sync_bn_forward_mean_impl, MUSA,
                     sync_bn_forward_mean_musa);
REGISTER_DEVICE_IMPL(sync_bn_forward_var_impl, MUSA, sync_bn_forward_var_musa);
REGISTER_DEVICE_IMPL(sync_bn_forward_output_impl, MUSA,
                     sync_bn_forward_output_musa);
REGISTER_DEVICE_IMPL(sync_bn_backward_param_impl, MUSA,
                     sync_bn_backward_param_musa);
REGISTER_DEVICE_IMPL(sync_bn_backward_data_impl, MUSA,
                     sync_bn_backward_data_musa);

void ThreeInterpolateForwardMUSAKernelLauncher(int b, int c, int m, int n,
                                               const Tensor points,
                                               const Tensor idx,
                                               const Tensor weight, Tensor out);

void ThreeInterpolateBackwardMUSAKernelLauncher(int b, int c, int n, int m,
                                                const Tensor grad_out,
                                                const Tensor idx,
                                                const Tensor weight,
                                                Tensor grad_points);

void three_interpolate_forward_musa(int b, int c, int m, int n,
                                    const Tensor points, const Tensor idx,
                                    const Tensor weight, Tensor out) {
  ThreeInterpolateForwardMUSAKernelLauncher(b, c, m, n, points, idx, weight,
                                            out);
};

void three_interpolate_backward_musa(int b, int c, int n, int m,
                                     const Tensor grad_out, const Tensor idx,
                                     const Tensor weight, Tensor grad_points) {
  ThreeInterpolateBackwardMUSAKernelLauncher(b, c, n, m, grad_out, idx, weight,
                                             grad_points);
};

void three_interpolate_forward_impl(int b, int c, int m, int n,
                                    const Tensor points, const Tensor idx,
                                    const Tensor weight, Tensor out);

void three_interpolate_backward_impl(int b, int c, int n, int m,
                                     const Tensor grad_out, const Tensor idx,
                                     const Tensor weight, Tensor grad_points);
REGISTER_DEVICE_IMPL(three_interpolate_forward_impl, MUSA,
                     three_interpolate_forward_musa);
REGISTER_DEVICE_IMPL(three_interpolate_backward_impl, MUSA,
                     three_interpolate_backward_musa);

void ThreeNNForwardMUSAKernelLauncher(int b, int n, int m, const Tensor unknown,
                                      const Tensor known, Tensor dist2,
                                      Tensor idx);

void three_nn_forward_musa(int b, int n, int m, const Tensor unknown,
                           const Tensor known, Tensor dist2, Tensor idx) {
  ThreeNNForwardMUSAKernelLauncher(b, n, m, unknown, known, dist2, idx);
};

void three_nn_forward_impl(int b, int n, int m, const Tensor unknown,
                           const Tensor known, Tensor dist2, Tensor idx);
REGISTER_DEVICE_IMPL(three_nn_forward_impl, MUSA, three_nn_forward_musa);

void TINShiftForwardMUSAKernelLauncher(Tensor input, Tensor shift,
                                       Tensor output);

void TINShiftBackwardMUSAKernelLauncher(Tensor grad_output, Tensor shift,
                                        Tensor grad_input);

void tin_shift_forward_musa(Tensor input, Tensor shift, Tensor output) {
  TINShiftForwardMUSAKernelLauncher(input, shift, output);
}

void tin_shift_backward_musa(Tensor grad_output, Tensor shift,
                             Tensor grad_input) {
  TINShiftBackwardMUSAKernelLauncher(grad_output, shift, grad_input);
}

void tin_shift_forward_impl(Tensor input, Tensor shift, Tensor output);
void tin_shift_backward_impl(Tensor grad_output, Tensor shift,
                             Tensor grad_input);
REGISTER_DEVICE_IMPL(tin_shift_forward_impl, MUSA, tin_shift_forward_musa);
REGISTER_DEVICE_IMPL(tin_shift_backward_impl, MUSA, tin_shift_backward_musa);

torch::Tensor upfirdn2d_op(torch::Tensor input, torch::Tensor filter, int upx,
                           int upy, int downx, int downy, int padx0, int padx1,
                           int pady0, int pady1, bool flip, float gain);

torch::Tensor upfirdn2d_op_impl(torch::Tensor input, torch::Tensor filter,
                                int upx, int upy, int downx, int downy,
                                int padx0, int padx1, int pady0, int pady1,
                                bool flip, float gain);
REGISTER_DEVICE_IMPL(upfirdn2d_op_impl, MUSA, upfirdn2d_op);

int HardVoxelizeForwardMUSAKernelLauncher(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim = 3);

int NondeterministicHardVoxelizeForwardMUSAKernelLauncher(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim = 3);

void DynamicVoxelizeForwardMUSAKernelLauncher(
    const at::Tensor &points, at::Tensor &coors,
    const std::vector<float> voxel_size, const std::vector<float> coors_range,
    const int NDim = 3);

int hard_voxelize_forward_musa(const at::Tensor &points, at::Tensor &voxels,
                               at::Tensor &coors,
                               at::Tensor &num_points_per_voxel,
                               const std::vector<float> voxel_size,
                               const std::vector<float> coors_range,
                               const int max_points, const int max_voxels,
                               const int NDim) {
  return HardVoxelizeForwardMUSAKernelLauncher(
      points, voxels, coors, num_points_per_voxel, voxel_size, coors_range,
      max_points, max_voxels, NDim);
};

int nondeterministic_hard_voxelize_forward_musa(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim) {
  return NondeterministicHardVoxelizeForwardMUSAKernelLauncher(
      points, voxels, coors, num_points_per_voxel, voxel_size, coors_range,
      max_points, max_voxels, NDim);
};

void dynamic_voxelize_forward_musa(const at::Tensor &points, at::Tensor &coors,
                                   const std::vector<float> voxel_size,
                                   const std::vector<float> coors_range,
                                   const int NDim) {
  DynamicVoxelizeForwardMUSAKernelLauncher(points, coors, voxel_size,
                                           coors_range, NDim);
};

int hard_voxelize_forward_impl(const at::Tensor &points, at::Tensor &voxels,
                               at::Tensor &coors,
                               at::Tensor &num_points_per_voxel,
                               const std::vector<float> voxel_size,
                               const std::vector<float> coors_range,
                               const int max_points, const int max_voxels,
                               const int NDim);

int nondeterministic_hard_voxelize_forward_impl(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim);

void dynamic_voxelize_forward_impl(const at::Tensor &points, at::Tensor &coors,
                                   const std::vector<float> voxel_size,
                                   const std::vector<float> coors_range,
                                   const int NDim);

REGISTER_DEVICE_IMPL(hard_voxelize_forward_impl, MUSA,
                     hard_voxelize_forward_musa);
REGISTER_DEVICE_IMPL(nondeterministic_hard_voxelize_forward_impl, MUSA,
                     nondeterministic_hard_voxelize_forward_musa);
REGISTER_DEVICE_IMPL(dynamic_voxelize_forward_impl, MUSA,
                     dynamic_voxelize_forward_musa);

void RotatedFeatureAlignForwardMUSAKernelLauncher(const Tensor features,
                                                  const Tensor best_bboxes,
                                                  const float spatial_scale,
                                                  const int points,
                                                  Tensor output);

void RotatedFeatureAlignBackwardMUSAKernelLauncher(const Tensor top_grad,
                                                   const Tensor best_bboxes,
                                                   const float spatial_scale,
                                                   const int points,
                                                   Tensor bottom_grad);

void rotated_feature_align_forward_musa(const Tensor features,
                                        const Tensor best_bboxes,
                                        const float spatial_scale,
                                        const int points, Tensor output) {
  RotatedFeatureAlignForwardMUSAKernelLauncher(features, best_bboxes,
                                               spatial_scale, points, output);
};

void rotated_feature_align_backward_musa(const Tensor top_grad,
                                         const Tensor best_bboxes,
                                         const float spatial_scale,
                                         const int points, Tensor bottom_grad) {
  RotatedFeatureAlignBackwardMUSAKernelLauncher(
      top_grad, best_bboxes, spatial_scale, points, bottom_grad);
};

void rotated_feature_align_forward_impl(const Tensor features,
                                        const Tensor best_bboxes,
                                        const float spatial_scale,
                                        const int points, Tensor output);

void rotated_feature_align_backward_impl(const Tensor top_grad,
                                         const Tensor best_bboxes,
                                         const float spatial_scale,
                                         const int points, Tensor bottom_grad);

REGISTER_DEVICE_IMPL(rotated_feature_align_forward_impl, MUSA,
                     rotated_feature_align_forward_musa);
REGISTER_DEVICE_IMPL(rotated_feature_align_backward_impl, MUSA,
                     rotated_feature_align_backward_musa);

void PointsInPolygonsForwardMUSAKernelLauncher(const at::Tensor points,
                                               const at::Tensor polygons,
                                               const int rows, const int cols,
                                               at::Tensor output);

void points_in_polygons_forward_musa(const Tensor points, const Tensor polygons,
                                     Tensor output, const int rows,
                                     const int cols) {
  PointsInPolygonsForwardMUSAKernelLauncher(points, polygons, rows, cols,
                                            output);
};

void points_in_polygons_forward_impl(const Tensor points, const Tensor polygons,
                                     Tensor output, const int rows,
                                     const int cols);

REGISTER_DEVICE_IMPL(points_in_polygons_forward_impl, MUSA,
                     points_in_polygons_forward_musa);

torch::Tensor IndiceMaxpoolForwardMUSAKernelLauncher(torch::Tensor features,
                                                     torch::Tensor indicePairs,
                                                     torch::Tensor indiceNum,
                                                     int64_t numAct);

torch::Tensor indice_maxpool_forward_musa(torch::Tensor features,
                                          torch::Tensor indicePairs,
                                          torch::Tensor indiceNum,
                                          int64_t numAct) {
  return IndiceMaxpoolForwardMUSAKernelLauncher(features, indicePairs,
                                                indiceNum, numAct);
};

torch::Tensor indice_maxpool_forward_impl(torch::Tensor features,
                                          torch::Tensor indicePairs,
                                          torch::Tensor indiceNum,
                                          int64_t numAct);
REGISTER_DEVICE_IMPL(indice_maxpool_forward_impl, MUSA,
                     indice_maxpool_forward_musa);

torch::Tensor IndiceMaxpoolBackwardMUSAKernelLauncher(torch::Tensor features,
                                                      torch::Tensor outFeatures,
                                                      torch::Tensor outGrad,
                                                      torch::Tensor indicePairs,
                                                      torch::Tensor indiceNum);

torch::Tensor indice_maxpool_backward_musa(torch::Tensor features,
                                           torch::Tensor outFeatures,
                                           torch::Tensor outGrad,
                                           torch::Tensor indicePairs,
                                           torch::Tensor indiceNum) {
  return IndiceMaxpoolBackwardMUSAKernelLauncher(features, outFeatures, outGrad,
                                                 indicePairs, indiceNum);
};

torch::Tensor indice_maxpool_backward_impl(torch::Tensor features,
                                           torch::Tensor outFeatures,
                                           torch::Tensor outGrad,
                                           torch::Tensor indicePairs,
                                           torch::Tensor indiceNum);

REGISTER_DEVICE_IMPL(indice_maxpool_backward_impl, MUSA,
                     indice_maxpool_backward_musa)

torch::Tensor IndiceConvForwardMUSAKernelLauncher(
    torch::Tensor features, torch::Tensor filters, torch::Tensor indicePairs,
    torch::Tensor indiceNum, int64_t numActOut, int64_t _inverse,
    int64_t _subM);

torch::Tensor indice_conv_forward_musa(torch::Tensor features,
                                       torch::Tensor filters,
                                       torch::Tensor indicePairs,
                                       torch::Tensor indiceNum,
                                       int64_t numActOut, int64_t _inverse,
                                       int64_t _subM) {
  return IndiceConvForwardMUSAKernelLauncher(
      features, filters, indicePairs, indiceNum, numActOut, _inverse, _subM);
};

torch::Tensor indice_conv_forward_impl(torch::Tensor features,
                                       torch::Tensor filters,
                                       torch::Tensor indicePairs,
                                       torch::Tensor indiceNum,
                                       int64_t numActOut, int64_t _inverse,
                                       int64_t _subM);

REGISTER_DEVICE_IMPL(indice_conv_forward_impl, MUSA, indice_conv_forward_musa);

std::vector<torch::Tensor> IndiceConvBackwardMUSAKernelLauncher(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t _inverse,
    int64_t _subM);

std::vector<torch::Tensor> indice_conv_backward_musa(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t _inverse,
    int64_t _subM) {
  return IndiceConvBackwardMUSAKernelLauncher(
      features, filters, outGrad, indicePairs, indiceNum, _inverse, _subM);
};

std::vector<torch::Tensor> indice_conv_backward_impl(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t _inverse,
    int64_t _subM);

REGISTER_DEVICE_IMPL(indice_conv_backward_impl, MUSA,
                     indice_conv_backward_musa);

torch::Tensor FusedIndiceConvBatchnormMUSAKernelLauncher(
    torch::Tensor features, torch::Tensor filters, torch::Tensor bias,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numActOut,
    int64_t _inverse, int64_t _subM);

torch::Tensor fused_indice_conv_batchnorm_forward_musa(
    torch::Tensor features, torch::Tensor filters, torch::Tensor bias,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numActOut,
    int64_t _inverse, int64_t _subM) {
  return FusedIndiceConvBatchnormMUSAKernelLauncher(features, filters, bias,
                                                    indicePairs, indiceNum,
                                                    numActOut, _inverse, _subM);
};

torch::Tensor fused_indice_conv_batchnorm_forward_impl(
    torch::Tensor features, torch::Tensor filters, torch::Tensor bias,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numActOut,
    int64_t _inverse, int64_t _subM);

REGISTER_DEVICE_IMPL(fused_indice_conv_batchnorm_forward_impl, MUSA,
                     fused_indice_conv_batchnorm_forward_musa)

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

void ChamferDistanceForwardMUSAKernelLauncher(
    const Tensor xyz1, const Tensor xyz2, const Tensor dist1,
    const Tensor dist2, const Tensor idx1, const Tensor idx2);

void ChamferDistanceBackwardMUSAKernelLauncher(
    const Tensor xyz1, const Tensor xyz2, Tensor idx1, Tensor idx2,
    Tensor grad_dist1, Tensor grad_dist2, Tensor grad_xyz1, Tensor grad_xyz2);

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

void PrROIPoolForwardMUSAKernelLauncher(Tensor input, Tensor rois,
                                        Tensor output, int pooled_height,
                                        int pooled_width, float spatial_scale);

void PrROIPoolBackwardMUSAKernelLauncher(Tensor grad_output, Tensor rois,
                                         Tensor grad_input, int pooled_height,
                                         int pooled_width, float spatial_scale);

void PrROIPoolCoorBackwardMUSAKernelLauncher(
    Tensor output, Tensor grad_output, Tensor input, Tensor rois,
    Tensor grad_rois, int pooled_height, int pooled_width, float spatial_scale);

void prroi_pool_forward_musa(Tensor input, Tensor rois, Tensor output,
                             int pooled_height, int pooled_width,
                             float spatial_scale) {
  PrROIPoolForwardMUSAKernelLauncher(input, rois, output, pooled_height,
                                     pooled_width, spatial_scale);
}

void prroi_pool_backward_musa(Tensor grad_output, Tensor rois,
                              Tensor grad_input, int pooled_height,
                              int pooled_width, float spatial_scale) {
  PrROIPoolBackwardMUSAKernelLauncher(grad_output, rois, grad_input,
                                      pooled_height, pooled_width,
                                      spatial_scale);
}

void prroi_pool_coor_backward_musa(Tensor output, Tensor grad_output,
                                   Tensor input, Tensor rois, Tensor grad_rois,
                                   int pooled_height, int pooled_width,
                                   float spatial_scale) {
  PrROIPoolCoorBackwardMUSAKernelLauncher(output, grad_output, input, rois,
                                          grad_rois, pooled_height,
                                          pooled_width, spatial_scale);
}

void prroi_pool_forward_impl(Tensor input, Tensor rois, Tensor output,
                             int pooled_height, int pooled_width,
                             float spatial_scale);
void prroi_pool_backward_impl(Tensor grad_output, Tensor rois,
                              Tensor grad_input, int pooled_height,
                              int pooled_width, float spatial_scale);
void prroi_pool_coor_backward_impl(Tensor output, Tensor grad_output,
                                   Tensor input, Tensor rois, Tensor grad_rois,
                                   int pooled_height, int pooled_width,
                                   float spatial_scale);
REGISTER_DEVICE_IMPL(prroi_pool_forward_impl, MUSA, prroi_pool_forward_musa);
REGISTER_DEVICE_IMPL(prroi_pool_backward_impl, MUSA, prroi_pool_backward_musa);
REGISTER_DEVICE_IMPL(prroi_pool_coor_backward_impl, MUSA,
                     prroi_pool_coor_backward_musa);

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
