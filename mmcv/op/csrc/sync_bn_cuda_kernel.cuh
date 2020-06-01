#include <stdio.h>

const size_t cuda_num_threads = 1024;

template <typename scalar_t>
__global__ void forward_mean_before_reduce(const size_t num, const size_t channels, const size_t spatial_dim, const scalar_t *input, float *mean) {
    __shared__ float buffer[cuda_num_threads];
    const size_t tid = threadIdx.x;
    const size_t c = blockIdx.x;
    buffer[tid] = 0;
    for (size_t i = tid; i < num * spatial_dim; i += blockDim.x) {
        size_t index = i / spatial_dim * (spatial_dim * (channels - 1)) + i + blockIdx.x * spatial_dim;
        buffer[tid] += input[index];
    }
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            buffer[tid] += buffer[tid + s];
        }
        __syncthreads();
    }
    size_t total = num * spatial_dim;
    if (tid == 0) {
        mean[c] = buffer[0] / total;
    }
}

template <>
__global__ void forward_mean_before_reduce(const size_t num, const size_t channels, const size_t spatial_dim, const phalf *input, float *mean) {
    __shared__ float buffer[cuda_num_threads];
    const size_t tid = threadIdx.x;
    const size_t c = blockIdx.x;
    buffer[tid] = 0;
    for (size_t i = tid; i < num * spatial_dim; i += blockDim.x) {
        size_t index = i / spatial_dim * (spatial_dim * (channels - 1)) + i + blockIdx.x * spatial_dim;
        buffer[tid] += static_cast<float>(input[index]);
    }
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            buffer[tid] += buffer[tid + s];
        }
        __syncthreads();
    }
    size_t total = num * spatial_dim;
    if (tid == 0) {
        mean[c] = buffer[0] / total;
    }
}

template <typename scalar_t>
__global__ void forward_var_before_reduce(const size_t num, const size_t channels, const size_t spatial_dim, const scalar_t *input, const float *mean, float* var) {
    __shared__ float buffer[cuda_num_threads];
    const size_t tid = threadIdx.x;
    const size_t c = blockIdx.x;
    buffer[tid] = 0;
    for (size_t i = tid; i < num * spatial_dim; i += blockDim.x) {
        size_t index = i / spatial_dim * (spatial_dim * (channels - 1)) + i + blockIdx.x * spatial_dim;
        float td = input[index] - mean[c];
        buffer[tid] += td * td;
    }
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            buffer[tid] += buffer[tid + s];
        }
        __syncthreads();
    }
    size_t total = num * spatial_dim;
    if (tid == 0) {
        var[c] = buffer[0] / total;
    }
}

template <>
__global__ void forward_var_before_reduce(const size_t num, const size_t channels, const size_t spatial_dim, const phalf *input, const float *mean, float* var) {
    __shared__ float buffer[cuda_num_threads];
    const size_t tid = threadIdx.x;
    const size_t c = blockIdx.x;
    buffer[tid] = 0;
    for (size_t i = tid; i < num * spatial_dim; i += blockDim.x) {
        size_t index = i / spatial_dim * (spatial_dim * (channels - 1)) + i + blockIdx.x * spatial_dim;
        float td = static_cast<float>(input[index]) - mean[c];
        buffer[tid] += td * td;
    }
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            buffer[tid] += buffer[tid + s];
        }
        __syncthreads();
    }
    size_t total = num * spatial_dim;
    if (tid == 0) {
        var[c] = buffer[0] / total;
    }
}

template <typename scalar_t>
__global__ void forward_var_after_reduce(const size_t num, const size_t channels, const size_t spatial_dim, const size_t group_size, const scalar_t *input,
                                         const float eps, const float momentum, const float *mean, const float *var, float *running_mean, float *running_var,
                                         const float *weight, const float *bias, float *std, scalar_t *output) {
    float temp = sqrt(var[blockIdx.x] + eps);
    float weight_value = weight[blockIdx.x];
    float bias_value = bias[blockIdx.x];
    for(size_t i = threadIdx.x; i < num * spatial_dim; i += blockDim.x) {
      size_t location = i / spatial_dim * spatial_dim * channels + (i % spatial_dim) + blockIdx.x * spatial_dim;
      output[location] = (input[location] - mean[blockIdx.x]) / temp * weight_value + bias_value;
    }
    if(threadIdx.x == 0) {
      running_mean[blockIdx.x] = momentum * mean[blockIdx.x] + (1 - momentum) * running_mean[blockIdx.x];
      size_t count = num * spatial_dim * group_size;
      float var_unbias = count > 1 ? var[blockIdx.x] * count / (count - 1) : var[blockIdx.x];
      running_var[blockIdx.x] = momentum * var_unbias + (1 - momentum) * running_var[blockIdx.x];
      std[blockIdx.x] = temp;
    }
}

template <>
__global__ void forward_var_after_reduce(const size_t num, const size_t channels, const size_t spatial_dim, const size_t group_size, const phalf *input,
                                         const float eps, const float momentum, const float *mean, const float *var, float *running_mean, float *running_var,
                                         const float *weight, const float *bias, float *std, phalf *output) {
    float temp = sqrt(var[blockIdx.x] + eps);
    float weight_value = weight[blockIdx.x];
    float bias_value = bias[blockIdx.x];
    for(size_t i = threadIdx.x; i < num * spatial_dim; i += blockDim.x) {
      size_t location = i / spatial_dim * spatial_dim * channels + (i % spatial_dim) + blockIdx.x * spatial_dim;
      output[location] = static_cast<phalf>((static_cast<float>(input[location]) - mean[blockIdx.x]) / temp * weight_value + bias_value);
    }
    if(threadIdx.x == 0) {
      running_mean[blockIdx.x] = momentum * mean[blockIdx.x] + (1 - momentum) * running_mean[blockIdx.x];
      size_t count = num * spatial_dim * group_size;
      float var_unbias = count > 1 ? var[blockIdx.x] * count / (count - 1) : var[blockIdx.x];
      running_var[blockIdx.x] = momentum * var_unbias + (1 - momentum) * running_var[blockIdx.x];
      std[blockIdx.x] = temp;
    }
}

template <typename scalar_t>
__global__ void backward_param_kernel(const size_t num, const size_t channels, const size_t spatial_dim, const scalar_t *input, 
                                      const float *mean, float *weight_diff, float *bias_diff, const float *std, const scalar_t *grad_output) {
    __shared__ float buffer1[cuda_num_threads];
    __shared__ float buffer2[cuda_num_threads];

    const size_t tid = threadIdx.x;
    const size_t c = blockIdx.x;

    buffer1[tid] = buffer2[tid] = 0;
    for (size_t i = tid; i < num * spatial_dim; i += blockDim.x) {
        const size_t index = i / spatial_dim * channels * spatial_dim + c * spatial_dim + i % spatial_dim;
        buffer1[tid] += grad_output[index] * ((input[index] - mean[c]) / std[c]);
        buffer2[tid] += grad_output[index];
    }
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            buffer1[tid] += buffer1[tid + s];
            buffer2[tid] += buffer2[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        weight_diff[c] = buffer1[0];
        bias_diff[c] = buffer2[0];
    }
}


template <>
__global__ void backward_param_kernel(const size_t num, const size_t channels, const size_t spatial_dim, const phalf *input, 
                                      const float *mean, float *weight_diff, float *bias_diff, const float *std, const phalf *grad_output) {
    __shared__ float buffer1[cuda_num_threads];
    __shared__ float buffer2[cuda_num_threads];

    const size_t tid = threadIdx.x;
    const size_t c = blockIdx.x;

    buffer1[tid] = buffer2[tid] = 0;
    for (size_t i = tid; i < num * spatial_dim; i += blockDim.x) {
        const size_t index = i / spatial_dim * channels * spatial_dim + c * spatial_dim + i % spatial_dim;
        buffer1[tid] += static_cast<float>(grad_output[index]) * ((static_cast<float>(input[index]) - mean[c]) / std[c]);
        buffer2[tid] += static_cast<float>(grad_output[index]);
    }
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            buffer1[tid] += buffer1[tid + s];
            buffer2[tid] += buffer2[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        weight_diff[c] = buffer1[0];
        bias_diff[c] = buffer2[0];
    }
}

template <typename scalar_t>
__global__ void backward_data_kernel(const size_t num, const size_t channels, const size_t spatial_dim, const scalar_t *input,
                                     scalar_t *grad_input, const float *mean, const float *weight, const float *weight_diff, const float *bias_diff,
                                     const float *std, const scalar_t *grad_output) {
    size_t factor = num * spatial_dim;
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num * channels * spatial_dim; i += blockDim.x * gridDim.x) {
        size_t c = (i / spatial_dim) % channels;
        grad_input[i] = weight[c] * (grad_output[i] - (weight_diff[c] * (input[i] - mean[c]) / std[c] + bias_diff[c]) / factor) / std[c];
    }
}

template <>
__global__ void backward_data_kernel(const size_t num, const size_t channels, const size_t spatial_dim, const phalf *input,
                                     phalf *grad_input, const float *mean, const float *weight, const float *weight_diff, const float *bias_diff,
                                     const float *std, const phalf *grad_output) {
    size_t factor = num * spatial_dim;
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num * channels * spatial_dim; i += blockDim.x * gridDim.x) {
        size_t c = (i / spatial_dim) % channels;
        grad_input[i] = static_cast<phalf>(weight[c] * (static_cast<float>(grad_output[i]) - (weight_diff[c] * (static_cast<float>(input[i]) - mean[c]) / std[c] + bias_diff[c]) / factor) / std[c]);
    }
}
