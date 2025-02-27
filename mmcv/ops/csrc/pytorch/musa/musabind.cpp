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
