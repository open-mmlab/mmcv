#include "pytorch_npu_helper.hpp"
#include "add_bisheng.hpp"

void test_add_impl_npu(const Tensor input1, const Tensor input2, Tensor output) {
    bscpp_add_launch(input1, input2, output);
}

void test_add_forward_impl(const Tensor input1, const Tensor input2, Tensor output);

REGISTER_NPU_IMPL(test_add_forward_impl, test_add_impl_npu);