#include <stdio.h>
#include <float.h>
#include <parrots/extension.hpp>
#include "parrots_cuda_helper.hpp"
using phalf=float16;
#include "sync_bn_cuda_kernel.cuh"

using namespace parrots;

void cudaSyncBNForwardStep1(size_t n, size_t c, size_t h, size_t w,
                            const DArrayLite input, DArrayLite mean, cudaStream_t stream) {
    PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.elemType().prim(), ([&] {
                const scalar_t *input_ptr = input.ptr<scalar_t>();
                float *mean_ptr = mean.ptr<float>();
                forward_mean_before_reduce<scalar_t><<<c, cuda_num_threads, 0, stream>>>(
                    n, c, h * w, input_ptr, mean_ptr);
            }));

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSyncBNForwardStep1 forward_local_stats_kernel failed : %s (%zu, %zu, %zu, %zu)\n",
                cudaGetErrorString(err), n, c, h, w);
        exit(-1);
    }
}

void cudaSyncBNForwardStep2(size_t n, size_t c, size_t h, size_t w, const DArrayLite input,
                            const DArrayLite mean, DArrayLite var, cudaStream_t stream) {
    PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.elemType().prim(),  ([&] {
                const scalar_t *input_ptr = input.ptr<scalar_t>();
                const float *mean_ptr = mean.ptr<float>();
                float *var_ptr = var.ptr<float>();
                forward_var_before_reduce<scalar_t><<<c, cuda_num_threads, 0, stream>>>(
                    n, c, h * w, input_ptr, mean_ptr, var_ptr);
            }));

    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaSyncBNForwardStep2 forward_var_before_reduce failed : %s (%zu, %zu, %zu, %zu)\n", cudaGetErrorString( err ), n, c, h, w );
        exit( -1 );
    }
}

void cudaSyncBNForwardStep3(size_t n, size_t c, size_t h, size_t w, size_t group_size, const DArrayLite input,
                            const float eps, const float momentum, const DArrayLite mean, const DArrayLite var,
                            DArrayLite running_mean, DArrayLite running_var, const DArrayLite weight,
                            const DArrayLite bias, DArrayLite std, DArrayLite output, cudaStream_t stream) {
    PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.elemType().prim(), ([&] {
                const scalar_t *input_ptr = input.ptr<scalar_t>();
                const float *mean_ptr = mean.ptr<float>();
                const float *var_ptr = var.ptr<float>();
                float *running_mean_ptr = running_mean.ptr<float>();
                float *running_var_ptr = running_var.ptr<float>();
                const float *weight_ptr = weight.ptr<float>();
                const float *bias_ptr = bias.ptr<float>();
                float *std_ptr = std.ptr<float>();
                scalar_t *output_ptr = output.ptr<scalar_t>();
                forward_var_after_reduce<scalar_t><<<c, cuda_num_threads, 0, stream>>>(
                    n, c, h*w, group_size, input_ptr, eps, momentum, mean_ptr, var_ptr,
                    running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, std_ptr, output_ptr);
            }));

    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaSyncBNForwardStep3 forward_var_after_reduce failed : %s (%zu, %zu, %zu, %zu)\n", cudaGetErrorString( err ), n, c, h, w );
        exit( -1 );
    }
}

void cudaSyncBNBackwardStep1(size_t n, size_t c, size_t h, size_t w, const DArrayLite input,
                             const DArrayLite mean, DArrayLite weight_diff, DArrayLite bias_diff,
                             const DArrayLite std, const DArrayLite grad_output, cudaStream_t stream) {

    PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.elemType().prim(), ([&] {
                const scalar_t *input_ptr = input.ptr<scalar_t>();
                const float *mean_ptr = mean.ptr<float>();
                float *weight_diff_ptr = weight_diff.ptr<float>();
                float *bias_diff_ptr = bias_diff.ptr<float>();
                const float *std_ptr = std.ptr<float>();
                const scalar_t *grad_output_ptr = grad_output.ptr<scalar_t>();
                backward_param_kernel<scalar_t> <<<c, cuda_num_threads, 0, stream>>>(
                    n, c, h*w, input_ptr, mean_ptr, weight_diff_ptr,
                    bias_diff_ptr, std_ptr, grad_output_ptr);
            }));

    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaSyncBNBackwardStep1 backward_local_stats_kernel failed : %s (%zu, %zu, %zu, %zu)\n", cudaGetErrorString( err ), n, c, h, w );
        exit( -1 );
    }
}

void cudaSyncBNBackwardStep2(size_t n, size_t c, size_t h, size_t w, const DArrayLite input,
                            DArrayLite grad_input, const DArrayLite mean, const DArrayLite weight,
                            const DArrayLite weight_diff, const DArrayLite bias_diff, const DArrayLite std,
                            const DArrayLite grad_output, cudaStream_t stream) {
    const int blockSize = 1024;
    const size_t gridSize = (w*h*c*n + blockSize - 1) / blockSize;
    PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.elemType().prim(), ([&] {
                const scalar_t *input_ptr = input.ptr<scalar_t>();
                scalar_t *grad_input_ptr = grad_input.ptr<scalar_t>();
                const float *mean_ptr = mean.ptr<float>();
                const float *weight_ptr = weight.ptr<float>();
                const float *weight_diff_ptr = weight_diff.ptr<float>();
                const float *bias_diff_ptr = bias_diff.ptr<float>();
                const float *std_ptr = std.ptr<float>();
                const scalar_t *grad_output_ptr = grad_output.ptr<scalar_t>();
                backward_data_kernel<scalar_t><<<gridSize, blockSize, 0, stream>>>(
                    n, c, h*w, input_ptr, grad_input_ptr, mean_ptr, weight_ptr,
                    weight_diff_ptr, bias_diff_ptr, std_ptr, grad_output_ptr);
            }));

    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaSyncBNBackwardStep2 backward_compute_kernel failed : %s (%zu, %zu, %zu, %zu)\n", cudaGetErrorString( err ), n, c, h, w );
        exit( -1 );
    }
}
