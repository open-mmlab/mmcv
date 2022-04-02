// Copyright (c) OpenMMLab. All rights reserved
// Adapted from https://github.com/lilanxiao/Rotated_IoU/cuda_op/sort_vert.cpp  # noqa
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "pytorch_cpp_helper.hpp"

#define MAX_NUM_VERT_IDX 9

void sort_vertices_wrapper(int b, int n, int m, const float *vertices, const bool *mask, const int *num_valid, int* idx);

at::Tensor diff_iou_rotated_sort_vertices(at::Tensor vertices, at::Tensor mask, at::Tensor num_valid){
    CHECK_CONTIGUOUS(vertices);
    CHECK_CONTIGUOUS(mask);
    CHECK_CONTIGUOUS(num_valid);
    CHECK_CUDA(vertices);
    CHECK_CUDA(mask);
    CHECK_CUDA(num_valid);

    int b = vertices.size(0);
    int n = vertices.size(1);
    int m = vertices.size(2);
    at::Tensor idx = torch::zeros({b, n, MAX_NUM_VERT_IDX},
                        at::device(vertices.device()).dtype(at::ScalarType::Int));

    // fix issue with multi-gpu (kernel only works for cuda:0)
    const at::cuda::OptionalCUDAGuard device_guard(device_of(idx));

    sort_vertices_wrapper(b, n, m, vertices.data_ptr<float>(), mask.data_ptr<bool>(),
                         num_valid.data_ptr<int>(), idx.data_ptr<int>());

    return idx;
}
