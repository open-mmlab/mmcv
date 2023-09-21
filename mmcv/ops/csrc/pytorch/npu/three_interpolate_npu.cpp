#include <ATen/ATen.h>
#include <cstdio>
#include <iostream>
#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void three_interpolate_forward_npu(int b, int c, int m, int n
                                   const Tensor points,
                                   const Tensor idx,
                                   const Tensor weight,
                                   Tensor out)
{
    auto point_c_trans = points.transpose(1, 2);

    OpCommand cmd;
    cmd.Name("ThreeInterpolate")
        .Input(point_c_trans)
        .Input(idx)
        .Input(weight)
        .Output(out)
        .Run();

    auto output = out.view({b, n, c}).transpose(1, 2);
    auto res = NpuUtils::format_contiguous(output);
    out.copy_(res);
}

void three_interpolate_forward_impl(int b, int c, int m, int n,
                                    const Tensor points, const Tensor idx,
                                    const Tensor weight, Tensor out);

REGISTER_NPU_IMPL(three_interpolate_forward_impl, three_interpolate_forward_npu);