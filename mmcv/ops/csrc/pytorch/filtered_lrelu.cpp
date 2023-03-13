#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

std::tuple<torch::Tensor, torch::Tensor, int> filtered_lrelu_op_impl(
    torch::Tensor x, torch::Tensor fu, torch::Tensor fd, torch::Tensor b,
    torch::Tensor si, int up, int down, int px0, int px1, int py0, int py1,
    int sx, int sy, float gain, float slope, float clamp, bool flip_filters,
    bool writeSigns) {
  return DISPATCH_DEVICE_IMPL(filtered_lrelu_op_impl, x, fu, fd, b, si, up,
                              down, px0, px1, py0, py1, sx, sy, gain, slope,
                              clamp, flip_filters, writeSigns);
}

std::tuple<torch::Tensor, torch::Tensor, int> filtered_lrelu(
    torch::Tensor x, torch::Tensor fu, torch::Tensor fd, torch::Tensor b,
    torch::Tensor si, int up, int down, int px0, int px1, int py0, int py1,
    int sx, int sy, float gain, float slope, float clamp, bool flip_filters,
    bool writeSigns) {
  return filtered_lrelu_op_impl(x, fu, fd, b, si, up, down, px0, px1, py0, py1,
                                sx, sy, gain, slope, clamp, flip_filters,
                                writeSigns);
}

torch::Tensor filtered_lrelu_act_op_impl(torch::Tensor x, torch::Tensor si,
                                         int sx, int sy, float gain,
                                         float slope, float clamp,
                                         bool writeSigns) {
  return DISPATCH_DEVICE_IMPL(filtered_lrelu_act_op_impl, x, si, sx, sy, gain,
                              slope, clamp, writeSigns);
}

torch::Tensor filtered_lrelu_act_(torch::Tensor x, torch::Tensor si, int sx,
                                  int sy, float gain, float slope, float clamp,
                                  bool writeSigns) {
  return filtered_lrelu_act_op_impl(x, si, sx, sy, gain, slope, clamp,
                                    writeSigns);
}
