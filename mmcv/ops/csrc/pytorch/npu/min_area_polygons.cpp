#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void min_area_polygons_npu(const Tensor pointsets, Tensor polygons) {
  OpCommand cmd;
  cmd.Name("MinAreaPolygons")
      .Input(pointsets)
      .Output(polygons)
      .Run();
}

void min_area_polygons_impl(const Tensor pointsets, Tensor polygons);

REGISTER_NPU_IMPL(min_area_polygons_impl, min_area_polygons_npu);
