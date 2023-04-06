// Copyright(c) OpenMMLab.All rights reserved.
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

template <typename T>
void assign_score_withk_forward_cpu_kernel(int B, int N0, int N1, int M, int K,
                                           int O, int aggregate,
                                           const T *points, const T *centers,
                                           const T *scores,
                                           const int64_t *knn_idx, T *output) {
  int b, o, n, k, m;
  for (b = 0; b < B; b++) {
    for (n = 0; n < N1; n++) {
      int centerIndex = knn_idx[b * N1 * O + n * O];
      const T *c = centers + b * N0 * M * O + centerIndex * M * O;
      for (k = 0; k < K; k++) {
        int pointIndex = knn_idx[b * N1 * O + n * O + k];
        const T *p = points + b * N0 * M * O + pointIndex * M * O;
        for (m = 0; m < M; m++) {
          const T score = scores[b * N1 * K * M + n * K * M + k * M + m];
          for (o = 0; o < O; o++) {
            output[b * O * N1 * K + o * N1 * K + n * K + k] +=
                (p[m * O + o] - c[m * O + o]) * score;
          }
        }
      }
    }
  }
}

void AssignScoreWithKForwardCPULauncher(int B, int N0, int N1, int M, int K,
                                        int O, int aggregate,
                                        const Tensor &points,
                                        const Tensor &centers,
                                        const Tensor &scores,
                                        const Tensor &knn_idx, Tensor &output) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      points.scalar_type(), "assign_score_withk_forward_cpu_kernel", [&] {
        assign_score_withk_forward_cpu_kernel<scalar_t>(
            B, N0, N1, M, K, O, aggregate, points.data_ptr<scalar_t>(),
            centers.data_ptr<scalar_t>(), scores.data_ptr<scalar_t>(),
            knn_idx.data_ptr<int64_t>(), output.data_ptr<scalar_t>());
      });
}

void assign_score_withk_forward_cpu(int B, int N0, int N1, int M, int K, int O,
                                    int aggregate, const Tensor &points,
                                    const Tensor &centers, const Tensor &scores,
                                    const Tensor &knn_idx, Tensor &output) {
  AssignScoreWithKForwardCPULauncher(B, N0, N1, M, K, O, aggregate, points,
                                     centers, scores, knn_idx, output);
}

void assign_score_withk_forward_impl(int B, int N0, int N1, int M, int K, int O,
                                     int aggregate, const Tensor &points,
                                     const Tensor &centers,
                                     const Tensor &scores,
                                     const Tensor &knn_idx, Tensor &output);

REGISTER_DEVICE_IMPL(assign_score_withk_forward_impl, CPU,
                     assign_score_withk_forward_cpu);
