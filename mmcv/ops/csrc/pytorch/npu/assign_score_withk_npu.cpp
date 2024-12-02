#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void assign_score_withk_forward_npu(int B, int N0, int N1, int M, int K, int O,
                                    int aggregate, const Tensor& points,
                                    const Tensor& centers, const Tensor& scores,
                                    const Tensor& knn_idx, Tensor& output) {
  at::Tensor points_trans = points.permute({0, 3, 1, 2});
  at::Tensor centers_trans = centers.permute({0, 3, 1, 2});
  EXEC_NPU_CMD(aclnnAssignScoreWithk, points_trans, centers_trans, scores,
               knn_idx, B, N0, N1, M, K, O, aggregate, output);
}

void assign_score_withk_forward_impl(int B, int N0, int N1, int M, int K, int O,
                                     int aggregate, const Tensor& points,
                                     const Tensor& centers,
                                     const Tensor& scores,
                                     const Tensor& knn_idx, Tensor& output);

REGISTER_NPU_IMPL(assign_score_withk_forward_impl,
                  assign_score_withk_forward_npu);
