#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void active_rotated_filter_forward_impl(const Tensor input,
                                        const Tensor indices, Tensor output);

void active_rotated_filter_forward_npu(const Tensor input,
                                       const Tensor indices, Tensor output) {
  OpCommand cmd;
  cmd.Name("ActiveRotatedFilter")
      .Input(input)
      .Input(indices)
      .Output(output)
      .Run();
}

REGISTER_NPU_IMPL(active_rotated_filter_forward_impl, active_rotated_filter_forward_npu);
