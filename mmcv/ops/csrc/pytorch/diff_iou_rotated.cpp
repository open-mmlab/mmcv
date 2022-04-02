#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x)                                           \
    do {                                                        \
        TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor");   \
    } while (0)

#define CHECK_CONTIGUOUS(x)                                                     \
    do {                                                                        \
        TORCH_CHECK(x.is_contiguous(), #x " must ne a contiguous tensor");       \
    } while (0)                                                                 

#define CHECK_IS_INT(x)                                     \
    do {                                                    \
        TORCH_CHECK(x.scalar_type()==at::ScalarType::Int,   \
                    #x " must be a int tensor");            \
    } while (0)

#define CHECK_IS_FLOAT(x)                                       \
    do {                                                        \
        TORCH_CHECK(x.scalar_type()==at::ScalarType::Float,    \
                    #x " must be a float tensor");              \
    } while (0)                                                 

#define CHECK_IS_BOOL(x)                                       \
    do {                                                        \
        TORCH_CHECK(x.scalar_type()==at::ScalarType::Bool,    \
                    #x " must be a bool tensor");             \
    } while (0)

#define MAX_NUM_VERT_IDX 9

void sort_vertices_wrapper(int b, int n, int m, const float *vertices, const bool *mask, const int *num_valid, int* idx);

at::Tensor diff_iou_rotated_sort_vertices(at::Tensor vertices, at::Tensor mask, at::Tensor num_valid){
    CHECK_CONTIGUOUS(vertices);
    CHECK_CONTIGUOUS(mask);
    CHECK_CONTIGUOUS(num_valid);
    CHECK_CUDA(vertices);
    CHECK_CUDA(mask);
    CHECK_CUDA(num_valid);
    CHECK_IS_FLOAT(vertices);
    CHECK_IS_BOOL(mask);
    CHECK_IS_INT(num_valid);

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

