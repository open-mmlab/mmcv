#include "pytorch_npu_helper.hpp"
using namespace NPU_NAME_SPACE;
using namespace std;

Tensor diff_iou_rotated_sort_vertices_npu(Tensor vertices, Tensor mask,
                                          Tensor num_valid) {
  TORCH_CHECK(vertices.dim() == 4,
              "vertices must be a 4D Tensor, but got: ", vertices.dim());
  TORCH_CHECK(mask.dim() == 3,
              "mask must be a 3D Tensor, but got: ", mask.dim());
  TORCH_CHECK(num_valid.dim() == 2,
              "num_valid must be a 2D Tensor, but got: ", num_valid.dim());

  uint32_t B = vertices.size(0);
  uint32_t N = vertices.size(1);

  at::Tensor sortedIdx = at::empty({B, N, 9}, num_valid.options());
  at::Tensor mask_fp = mask.to(at::kFloat);

  EXEC_NPU_CMD(aclnnDiffIouRotatedSortVertices, vertices, mask_fp, num_valid,
               sortedIdx);

  return sortedIdx;
}

Tensor diff_iou_rotated_sort_vertices_forward_impl(Tensor vertices, Tensor mask,
                                                   Tensor num_valid);

REGISTER_NPU_IMPL(diff_iou_rotated_sort_vertices_forward_impl,
                  diff_iou_rotated_sort_vertices_npu);
