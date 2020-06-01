#include <ATen/ATen.h>
#include "pytorch_cuda_helper.hpp"
using phalf=at::Half;
#include "sync_bn_cuda_kernel.cuh"

using namespace at;

void cudaSyncBNForwardStep1(size_t n, size_t c, size_t h, size_t w,
                            const at::Tensor input, at::Tensor mean) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(), "cudaSyncBNForwardStep1", ([&] {
                const scalar_t *input_ptr = input.data_ptr<scalar_t>();
                float *mean_ptr = mean.data_ptr<float>();
                forward_mean_before_reduce<scalar_t><<<c, cuda_num_threads>>>(n, c, h * w, input_ptr, mean_ptr);
            }));

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSyncBNForwardStep1 forward_local_stats_kernel failed : %s (%zu, %zu, %zu, %zu)\n",
                cudaGetErrorString(err), n, c, h, w);
        exit(-1);
    }
}

void cudaSyncBNForwardStep2(size_t n, size_t c, size_t h, size_t w, const at::Tensor input,
                            const at::Tensor mean, at::Tensor var) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(), "cudaSyncBNForwardStep2", ([&] {
                const scalar_t *input_ptr = input.data_ptr<scalar_t>();
                const float *mean_ptr = mean.data_ptr<float>();
                float *var_ptr = var.data_ptr<float>();
                forward_var_before_reduce<scalar_t><<<c, cuda_num_threads>>>(n, c, h * w, input_ptr, mean_ptr, var_ptr);
            }));

    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaSyncBNForwardStep2 forward_var_before_reduce failed : %s (%zu, %zu, %zu, %zu)\n", cudaGetErrorString( err ), n, c, h, w );
        exit( -1 );
    }
}

void cudaSyncBNForwardStep3(size_t n, size_t c, size_t h, size_t w, size_t group_size, const at::Tensor input,
                            const float eps, const float momentum, const at::Tensor mean, const at::Tensor var,
                            at::Tensor running_mean, at::Tensor running_var, const at::Tensor weight,
                            const at::Tensor bias, at::Tensor std, at::Tensor output) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(), "cudaSyncBNForwardStep3", ([&] {
                const scalar_t *input_ptr = input.data_ptr<scalar_t>();
                const float *mean_ptr = mean.data_ptr<float>();
                const float *var_ptr = var.data_ptr<float>();
                float *running_mean_ptr = running_mean.data_ptr<float>();
                float *running_var_ptr = running_var.data_ptr<float>();
                const float *weight_ptr = weight.data_ptr<float>();
                const float *bias_ptr = bias.data_ptr<float>();
                float *std_ptr = std.data_ptr<float>();
                scalar_t *output_ptr = output.data_ptr<scalar_t>();
                forward_var_after_reduce<scalar_t><<<c, cuda_num_threads>>>(n, c, h*w, group_size, input_ptr, eps, momentum, mean_ptr, var_ptr,
                                                                            running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
                                                                            std_ptr, output_ptr);
            }));

    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaSyncBNForwardStep3 forward_var_after_reduce failed : %s (%zu, %zu, %zu, %zu)\n", cudaGetErrorString( err ), n, c, h, w );
        exit( -1 );
    }
}

void cudaSyncBNBackwardStep1(size_t n, size_t c, size_t h, size_t w, const at::Tensor input,
                             const at::Tensor mean, at::Tensor weight_diff, at::Tensor bias_diff,
                             const at::Tensor std, const at::Tensor grad_output) {

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(), "cudaSyncBNBackwardStep1", ([&] {
                const scalar_t *input_ptr = input.data_ptr<scalar_t>();
                const float *mean_ptr = mean.data_ptr<float>();
                float *weight_diff_ptr = weight_diff.data_ptr<float>();
                float *bias_diff_ptr = bias_diff.data_ptr<float>();
                const float *std_ptr = std.data_ptr<float>();
                const scalar_t *grad_output_ptr = grad_output.data_ptr<scalar_t>();
                backward_param_kernel<scalar_t> <<<c, cuda_num_threads>>>(n, c, h*w, input_ptr, mean_ptr, weight_diff_ptr,
                                                                                bias_diff_ptr, std_ptr, grad_output_ptr);
            }));

    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaSyncBNBackwardStep1 backward_local_stats_kernel failed : %s (%zu, %zu, %zu, %zu)\n", cudaGetErrorString( err ), n, c, h, w );
        exit( -1 );
    }
}

void cudaSyncBNBackwardStep2(size_t n, size_t c, size_t h, size_t w, const at::Tensor input,
                            at::Tensor grad_input, const at::Tensor mean, const at::Tensor weight,
                            const at::Tensor weight_diff, const at::Tensor bias_diff, const at::Tensor std,
                            const at::Tensor grad_output) {
    const int blockSize = 1024;
    const size_t gridSize = (w*h*c*n + blockSize - 1) / blockSize;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(), "cudaSyncBNBackwardStep2", ([&] {
                const scalar_t *input_ptr = input.data_ptr<scalar_t>();
                scalar_t *grad_input_ptr = grad_input.data_ptr<scalar_t>();
                const float *mean_ptr = mean.data_ptr<float>();
                const float *weight_ptr = weight.data_ptr<float>();
                const float *weight_diff_ptr = weight_diff.data_ptr<float>();
                const float *bias_diff_ptr = bias_diff.data_ptr<float>();
                const float *std_ptr = std.data_ptr<float>();
                const scalar_t *grad_output_ptr = grad_output.data_ptr<scalar_t>();
                backward_data_kernel<scalar_t><<<gridSize, blockSize>>>(n, c, h*w, input_ptr, grad_input_ptr, mean_ptr, weight_ptr,
                                                                        weight_diff_ptr, bias_diff_ptr, std_ptr, grad_output_ptr);
            }));

    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaSyncBNBackwardStep2 backward_compute_kernel failed : %s (%zu, %zu, %zu, %zu)\n", cudaGetErrorString( err ), n, c, h, w );
        exit( -1 );
    }
}
