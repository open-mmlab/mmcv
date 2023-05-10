#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void active_rotated_filter_forward_impl(const Tensor input,
                                        const Tensor indices, Tensor output);

void active_rotated_filter_backward_impl(const Tensor grad_out,
                                         const Tensor indices, Tensor grad_in);

void active_rotated_filter_forward_npu(const Tensor input, const Tensor indices,
                                       Tensor output) {
  OpCommand cmd;
  cmd.Name("ActiveRotatedFilter")
      .Input(input)
      .Input(indices)
      .Output(output)
      .Run();
}

void active_rotated_filter_backward_npu(const Tensor grad_out,
                                        const Tensor indices, Tensor grad_in) {
  OpCommand cmd;
  cmd.Name("ActiveRotatedFilterGrad")
      .Input(grad_out)
      .Input(indices)
      .Output(grad_in)
      .Run();
}

REGISTER_NPU_IMPL(active_rotated_filter_forward_impl,
                  active_rotated_filter_forward_npu);

REGISTER_NPU_IMPL(active_rotated_filter_backward_impl,
                  active_rotated_filter_backward_npu);
