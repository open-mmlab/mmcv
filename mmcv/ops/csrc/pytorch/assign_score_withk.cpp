// Modified from
// https://github.com/CVMI-Lab/PAConv/tree/main/scene_seg/lib/paconv_lib/src/gpu
#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void AssignScoreWithKForwardCUDAKernelLauncher(
    int B, int N0, int N1, int M, int K, int O, int aggregate,
    const Tensor& points, const Tensor& centers, const Tensor& scores,
    const Tensor& knn_idx, Tensor& output);

void assign_score_withk_forward_cuda(int B, int N0, int N1, int M, int K, int O,
                                     int aggregate, const Tensor& points,
                                     const Tensor& centers,
                                     const Tensor& scores,
                                     const Tensor& knn_idx, Tensor& output) {
  AssignScoreWithKForwardCUDAKernelLauncher(
      B, N0, N1, M, K, O, aggregate, points, centers, scores, knn_idx, output);
};

void AssignScoreWithKBackwardCUDAKernelLauncher(
    int B, int N0, int N1, int M, int K, int O, int aggregate,
    const Tensor& grad_out, const Tensor& points, const Tensor& centers,
    const Tensor& scores, const Tensor& knn_idx, Tensor& grad_points,
    Tensor& grad_centers, Tensor& grad_scores);

void assign_score_withk_backward_cuda(
    int B, int N0, int N1, int M, int K, int O, int aggregate,
    const Tensor& grad_out, const Tensor& points, const Tensor& centers,
    const Tensor& scores, const Tensor& knn_idx, Tensor& grad_points,
    Tensor& grad_centers, Tensor& grad_scores) {
  AssignScoreWithKBackwardCUDAKernelLauncher(
      B, N0, N1, M, K, O, aggregate, grad_out, points, centers, scores, knn_idx,
      grad_points, grad_centers, grad_scores);
};
#endif

void assign_score_withk_forward(const Tensor& points, const Tensor& centers,
                                const Tensor& scores, const Tensor& knn_idx,
                                Tensor& output, int B, int N0, int N1, int M,
                                int K, int O, int aggregate) {
  if (points.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(centers);
    CHECK_CONTIGUOUS(scores);
    CHECK_CONTIGUOUS(knn_idx);
    CHECK_CONTIGUOUS(output);

    assign_score_withk_forward_cuda(B, N0, N1, M, K, O, aggregate, points,
                                    centers, scores, knn_idx, output);
#else
    AT_ERROR("assign_score_withk is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("assign_score_withk is not implemented on CPU");
  }
}

void assign_score_withk_backward(const Tensor& grad_out, const Tensor& points,
                                 const Tensor& centers, const Tensor& scores,
                                 const Tensor& knn_idx, Tensor& grad_points,
                                 Tensor& grad_centers, Tensor& grad_scores,
                                 int B, int N0, int N1, int M, int K, int O,
                                 int aggregate) {
  if (grad_points.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(scores);
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(centers);
    CHECK_CONTIGUOUS(knn_idx);
    CHECK_CONTIGUOUS(grad_scores);
    CHECK_CONTIGUOUS(grad_points);
    CHECK_CONTIGUOUS(grad_centers);

    assign_score_withk_backward_cuda(B, N0, N1, M, K, O, aggregate, grad_out,
                                     points, centers, scores, knn_idx,
                                     grad_points, grad_centers, grad_scores);
#else
    AT_ERROR("assign_score_withk is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("assign_score_withk is not implemented on CPU");
  }
}
