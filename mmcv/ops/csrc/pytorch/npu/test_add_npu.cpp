#include <iostream>
#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;

void test_add_npu(const Tensor input1, const Tensor input2, Tensor output) {
    at::Scalar s = 1;
    OpPreparation::CheckOut(
        {input1, input2},
        output,
        input1);

    OpCommand cmd;
    cmd.Name("AxpyV2")
        .Input(input1)
        .Input(input2)
        .Input(s, input1.scalar_type())
        .Output(output)
        .Run();
}

void test_add_impl(const Tensor input1, const Tensor input2, Tensor output);
REGISTER_NPU_IMPL(test_add_impl, test_add_npu);