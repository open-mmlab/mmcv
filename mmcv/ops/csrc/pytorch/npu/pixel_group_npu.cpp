#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

vector<vector<float>> pixel_group_npu(Tensor score, Tensor mask,
                                      Tensor embedding, Tensor kernel_label,
                                      Tensor kernel_contour,
                                      int kernel_region_num,
                                      float distance_threshold) {
  TORCH_CHECK(score.dim() == 2,
              "score.dim() must be 2, but got: ", score.dim());
  TORCH_CHECK(mask.dim() == 2, "mask.dim() must be 2, but got: ", mask.dim());
  TORCH_CHECK(embedding.dim() == 3,
              "embedding.dim() must be 3, but got: ", embedding.dim());
  TORCH_CHECK(kernel_label.dim() == 2,
              "kernel_label.dim() must be 2, but got: ", kernel_label.dim());
  TORCH_CHECK(
      kernel_contour.dim() == 2,
      "kernel_contour.dim() must be 2, but got: ", kernel_contour.dim());

  auto label_size = kernel_label.sizes();
  auto height = label_size[0];
  auto width = label_size[1];

  c10::SmallVector<int64_t, 8> point_vector_size = {kernel_region_num, 2};
  c10::SmallVector<int64_t, 8> label_updated_size = {height, width};
  at::Tensor point_vector = at::zeros(point_vector_size, score.options());
  at::Tensor label_updated =
      at::empty(label_updated_size, kernel_label.options());

  EXEC_NPU_CMD(aclnnPixelGroup, score, mask, embedding, kernel_label,
               kernel_contour, kernel_region_num, distance_threshold,
               point_vector, label_updated);

  std::vector<std::vector<float>> pixel_assignment(kernel_region_num);
  at::Tensor point_vector_cpu = point_vector.to(at::kCPU);
  at::Tensor label_updated_cpu = label_updated.to(at::kCPU);

  for (int32_t l = 0; l < kernel_region_num; l++) {
    pixel_assignment[l].push_back(point_vector_cpu[l][0].item<float>());
    pixel_assignment[l].push_back(point_vector_cpu[l][1].item<float>());
    if (pixel_assignment[l][1] > 0) {
      pixel_assignment[l][0] /= pixel_assignment[l][1];
    }
    if (l > 0) {
      at::Tensor valid_mask = (label_updated_cpu == l);
      at::Tensor indices = at::nonzero(valid_mask);
      for (int32_t i = 0; i < indices.size(0); i++) {
        auto x = indices[i][0].item<int32_t>();
        auto y = indices[i][1].item<int32_t>();
        pixel_assignment[l].push_back(y);
        pixel_assignment[l].push_back(x);
      }
    }
  }
  return pixel_assignment;
}

vector<vector<float>> pixel_group_impl(Tensor score, Tensor mask,
                                       Tensor embedding, Tensor kernel_label,
                                       Tensor kernel_contour,
                                       int kernel_region_num,
                                       float distance_threshold);

REGISTER_NPU_IMPL(pixel_group_impl, pixel_group_npu);
