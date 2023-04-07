#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void test_add_forward_impl(const Tensor input1, const Tensor input2, Tensor output) {
    DISPATCH_DEVICE_IMPL(test_add_forward_impl, input1, input2, output);
}

void test_add_forward(const Tensor input1, const Tensor input2, Tensor output) {
    test_add_forward_impl(input1, input2, output);
}
