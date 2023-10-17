#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

constexpr int32_t MAX_POLYGONS_BATCH = 2800;

void points_in_polygons_npu(const Tensor points, Tensor polygons, Tensor output,
                            const int rows, const int cols) {
  TORCH_CHECK(
      (polygons.sizes()[0] <= MAX_POLYGONS_BATCH),
      "The batch of polygons tensor must be less than MAX_POLYGONS_BATCH");
  at::Tensor trans_polygons = polygons.transpose(0, 1);
  OpCommand cmd;
  at::Tensor new_trans_polygons = NpuUtils::format_contiguous(trans_polygons);
  cmd.Name("PointsInPolygons")
      .Input(points, (string) "points")
      .Input(new_trans_polygons, (string) "polygons")
      .Output(output)
      .Run();
}

void points_in_polygons_forward_impl(const Tensor points, Tensor polygons,
                                     Tensor output, const int rows,
                                     const int cols);

REGISTER_NPU_IMPL(points_in_polygons_forward_impl, points_in_polygons_npu);
