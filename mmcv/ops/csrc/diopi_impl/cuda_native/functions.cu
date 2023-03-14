/**************************************************************************************************
 * Copyright (c) 2022, SenseTime Inc.
 * License
 * Author
 *
 *************************************************************************************************/

#include <diopi/functions.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cuda.h>

// #include <ATen/ATen.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>

#include "helper.hpp"

#define dispatch_dtype(fun, dtype, gridSize, blockSize, stream, ...)                             \
    if (diopi_dtype_int32 == dtype) {                                                            \
        fun<int32_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                           \
    } else if (diopi_dtype_uint32 == dtype) {                                                    \
        fun<uint32_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                          \
    } else if (diopi_dtype_int16 == dtype) {                                                      \
        fun<int16_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                           \
    } else if (diopi_dtype_uint16 == dtype) {                                                     \
        fun<uint16_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                          \
    } else if (diopi_dtype_int8 == dtype) {                                                       \
        fun<int8_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                            \
    } else if (diopi_dtype_uint8 == dtype) {                                                      \
        fun<uint8_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                           \
    } else if (diopi_dtype_float32 == dtype) {                                                    \
        fun<float><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                             \
    } else if (diopi_dtype_float64 == dtype) {                                                    \
        fun<double><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                            \
    } else if (diopi_dtype_bool == dtype) {                                                       \
        fun<bool><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                              \
    } else {                                                                                     \
        fprintf(stderr, "%s:%s: %s<%s %d><<<%d,%d>>>(%s)", __FILE__, __FUNCTION__, #fun, #dtype, \
                dtype, gridSize, blockSize, #__VA_ARGS__);                                       \
        return diopiDtypeNotSupported;                                                           \
    }

#define dispatch_float_types_and_half(fun, dtype, gridSize, blockSize, stream, ...)                             \
    if (diopi_dtype_float32 == dtype) {                                                    \
        fun<float><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                             \
    } else if (diopi_dtype_float64 == dtype) {                                                    \
        fun<double><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                            \
    } else {                                                                                     \
        fprintf(stderr, "%s:%s: %s<%s %d><<<%d,%d>>>(%s)", __FILE__, __FUNCTION__, #fun, #dtype, \
                dtype, gridSize, blockSize, #__VA_ARGS__);                                       \
        return diopiDtypeNotSupported;                                                           \
    }

template<typename T> __global__
void vecAdd(const void* a, const void* b, void* c, const int numel, const T alpha) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    const T* A = static_cast<const T*>(a);
    const T* B = static_cast<const T*>(b);
    T* C = static_cast<T*>(c);
    if (id < numel) {
        C[id] = A[id] + alpha * B[id];
    }
}

template<typename T> __global__
void vecAddBroadcast(const void* a, const void* b, void* c, const int numel, const T alpha,
        const int64_t* stride1, const int64_t* stride2, const int64_t* outStride, const int len) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    const T* A = static_cast<const T*>(a);
    const T* B = static_cast<const T*>(b);
    T* C = static_cast<T*>(c);
    int size = id;
    size_t idxA = 0;
    size_t idxB = 0;
    if (id < numel) {
        for (int i = 0; i < len; ++i) {
            int tmp = size / outStride[i];
            idxA += tmp * stride1[i];
            idxB += tmp * stride2[i];
            size = size % outStride[i];
        }
        C[id] = A[idxA] + alpha * B[idxB];
    }
}

template<typename T> __global__
void vecAddScalar(const void* a, const T b, void* c, const int numel, const T alpha) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    const T* A = static_cast<const T*>(a);
    T* C = static_cast<T*>(c);
    if (id < numel) {
        C[id] = A[id] + alpha * b;
    }
}

bool compareShape(const diopiSize_t& size1, const diopiSize_t& size2) {
    if (size1.len == size2.len) {
        for (int i = 0; i < size1.len; ++i) {
            if (size1.data[i] != size2.data[i]) {
                return 0;
            }
        }
        return 1;
    }
    return 0;
}

void computeStride(const diopiSize_t& size1, const diopiSize_t& size2, diopiSize_t outSize,
        int64_t* stride1, int64_t* stride2) {
    int length = size1.len;
    int len = outSize.len;
    int64_t stride = 1;
    for (int i = 0; i < len; ++i) {
        stride1[i] = 0;
        stride2[i] = 0;
    }
    for (int i = 1; i < length + 1; ++i) {
        if (size1.data[length - i] == outSize.data[len - i]) {
            stride1[len - i] = stride;
            stride *= outSize.data[len - i];
        }
    }
    length = size2.len;
    stride = 1;
    for (int i = 1; i < length + 1; ++i) {
        if (size2.data[length - i] == outSize.data[len - i]) {
            stride2[len - i] = stride;
            stride *= outSize.data[len - i];
        }
    }
}

extern "C" diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    auto stream  = impl::cuda::getStream(ctx);
    auto trInput = impl::cuda::makeTensor(input);
    auto trOther = impl::cuda::makeTensor(other);
    auto trOut   = impl::cuda::makeTensor(out);

    int blockSize = 256;
    double coff = 0.0;
    if (trInput.dtype() <= 7) {
        coff = alpha->ival;
    } else {
        coff = alpha->fval;
    }
    diopiSize_t inShape = trInput.shape();
    diopiSize_t othShape = trOther.shape();
    int gridSize  = (trOut.numel() + blockSize - 1) / blockSize;
    if (compareShape(inShape, othShape)) {
        dispatch_dtype(vecAdd, trInput.dtype(), gridSize, blockSize, stream,
            trInput.data(), trOther.data(), trOut.data(), trInput.numel(), coff);
    } else {
        diopiSize_t outShape = trOut.shape();
        diopiSize_t outStrideHost = trOut.stride();
        int len = outShape.len;
        int64_t nbytes = len * sizeof(int64_t);

        std::vector<int64_t> inStrideHost(len);
        std::vector<int64_t> othStrideHost(len);
        auto inStride = impl::cuda::requiresBuffer(ctx, nbytes);
        auto othStride = impl::cuda::requiresBuffer(ctx, nbytes);
        auto outStride = impl::cuda::requiresBuffer(ctx, nbytes);

        computeStride(inShape, othShape, outShape, inStrideHost.data(), othStrideHost.data());
        cudaMemcpyAsync(inStride.data(), inStrideHost.data(), nbytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(othStride.data(), othStrideHost.data(), nbytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(outStride.data(), outStrideHost.data, nbytes, cudaMemcpyHostToDevice, stream);

        dispatch_dtype(vecAddBroadcast, trInput.dtype(), gridSize, blockSize, stream,
           trInput.data(), trOther.data(), trOut.data(), trOut.numel(), coff, static_cast<const int64_t*>(inStride.data()),
           static_cast<const int64_t*>(othStride.data()), static_cast<const int64_t*>(outStride.data()), len);
    }
    return diopiSuccess;
}

extern "C" diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    auto stream  = impl::cuda::getStream(ctx);
    auto trInput = impl::cuda::makeTensor(input);
    auto trOut   = impl::cuda::makeTensor(out);
    int blockSize = 256;
    double coff = 0.0;
    double otherVal = 0.0;
    if (trInput.dtype() <= 7) {
        coff = alpha->ival;
        otherVal = other->ival;
    } else {
        coff = alpha->fval;
        otherVal = other->fval;
    }
    int gridSize = (trInput.numel() + blockSize - 1) / blockSize;
    dispatch_dtype(vecAddScalar, trInput.dtype(), gridSize, blockSize, stream,
        trInput.data(), otherVal, trOut.data(), trInput.numel(), coff);
    return diopiSuccess;
}

template<typename T> __global__
void vecFill(void* a, const float value, const int numel) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    T* A = static_cast<T*>(a);
    if (id < numel) {
        A[id] = static_cast<T>(value);
    }
}

extern "C" diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
    auto stream = impl::cuda::getStream(ctx);
    auto tr = impl::cuda::makeTensor(input);

    diopiDevice_t device = tr.device();
    diopiDtype_t  dtype  = tr.dtype();
    int64_t       numel  = tr.numel();
    float val;
    if (value->stype <= 7) {
        val = value->ival;
    } else {
        val = value->fval;
    }
    if (diopi_host == device) {
        return diopiErrorOccurred;
    } else {
        int blockSize = 256;
        int gridSize  = (numel + blockSize - 1) / blockSize;
        dispatch_dtype(vecFill, dtype, gridSize, blockSize, stream, tr.data(), val, numel);
    }

    return diopiSuccess;
}


#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                             \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);   \
       i += blockDim.x * gridDim.x)                                 \
    for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); \
         j += blockDim.y * gridDim.y)

#define CUDA_2D_KERNEL_BLOCK_LOOP(i, n, j, m)          \
  for (size_t i = blockIdx.x; i < (n); i += gridDim.x) \
    for (size_t j = blockIdx.y; j < (m); j += gridDim.y)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK) {
  int optimal_block_num = (N + num_threads - 1) / num_threads;
  int max_block_num = 4096;
  return min(optimal_block_num, max_block_num);
}

template <typename T>
__device__ T bilinear_interpolate(const T* input, const int height,
                                  const int width, T y, T x,
                                  const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) return 0;

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = input[y_low * width + x_low];
  T v2 = input[y_low * width + x_high];
  T v3 = input[y_high * width + x_low];
  T v4 = input[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width, T y, T x, T& w1, T& w2, T& w3, T& w4,
    int& x_low, int& x_high, int& y_low, int& y_high,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = input[y_low * width + x_low];
  // T v2 = input[y_low * width + x_high];
  // T v3 = input[y_high * width + x_low];
  // T v4 = input[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

#define MAX_SHARED_SCALAR_T 6144  // 49152 / 8 = 6144

template <typename scalar_t>
__global__ void chamfer_distance_forward_cuda_kernel_diopi(int b, int n,
                                                    const void* xyz, int m,
                                                    const void* xyz2,
                                                     void* result,
                                                     int* result_i) {
  __shared__ scalar_t buf[MAX_SHARED_SCALAR_T];
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int k2 = 0; k2 < m; k2 += THREADS_PER_BLOCK) {
      int end_k = min(m, k2 + THREADS_PER_BLOCK) - k2;
      const scalar_t* xyz2_ = static_cast<const scalar_t*>(xyz2);
      for (int j = threadIdx.x; j < end_k * 2; j += blockDim.x) {
        buf[j] = xyz2_[(i * m + k2) * 2 + j];
      }
      __syncthreads();
      const scalar_t* xyz_ = static_cast<const scalar_t*>(xyz);
      for (int j = threadIdx.x; j < n; j += blockDim.x * gridDim.y) {
        scalar_t x1 = xyz_[(i * n + j) * 2 + 0];
        scalar_t y1 = xyz_[(i * n + j) * 2 + 1];
        int best_i = 0;
        scalar_t best = 1e10;
        int end_ka = end_k & (~2);
        if (end_ka == THREADS_PER_BLOCK) {
          for (int k = 0; k < THREADS_PER_BLOCK; k += 4) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              scalar_t x2 = buf[(k + j) * 2] - x1;
              scalar_t y2 = buf[(k + j) * 2 + 1] - y1;
              scalar_t d = x2 * x2 + y2 * y2;
              if (d < best) {
                best = d;
                best_i = k + k2 + j;
              }
            }
          }
        } else {
          for (int k = 0; k < end_ka; k += 4) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              scalar_t x2 = buf[(k + j) * 2] - x1;
              scalar_t y2 = buf[(k + j) * 2 + 1] - y1;
              scalar_t d = x2 * x2 + y2 * y2;
              if (d < best) {
                best = d;
                best_i = k + k2 + j;
              }
            }
          }
        }
        for (int k = end_ka; k < end_k; k++) {
          scalar_t x2 = buf[k * 2 + 0] - x1;
          scalar_t y2 = buf[k * 2 + 1] - y1;
          scalar_t d = x2 * x2 + y2 * y2;
          if (k == 0 || d < best) {
            best = d;
            best_i = k + k2;
          }
        }
        scalar_t* result_ = static_cast<scalar_t*>(result);
        if (k2 == 0 || result_[(i * n + j)] > best) {
          result_[(i * n + j)] = best;
          result_i[(i * n + j)] = best_i;
        }
      }
      __syncthreads();
    }
  }
}

template <typename scalar_t>
__global__ void chamfer_distance_backward_cuda_kernel_diopi(
    int b, int n, const void* xyz1, int m, const void* xyz2,
    const void* grad_dist1, const int* idx1, void* grad_xyz1,
    void* grad_xyz2) {
  // const scalar_t* xyz1_ = static_cast<const scalar_t*>(xyz1);
  // const scalar_t* xyz2_ = static_cast<const scalar_t*>(xyz2);
  // const scalar_t* grad_dist1_ = static_cast<const scalar_t*>(grad_dist1);
  // scalar_t* grad_xyz1_ = static_cast<scalar_t*>(grad_xyz1);
  // scalar_t* grad_xyz2_ = static_cast<scalar_t*>(grad_xyz2);
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int j = threadIdx.x; j < n; j += blockDim.x * gridDim.y) {
      const scalar_t* xyz1_ = static_cast<const scalar_t*>(xyz1);
      scalar_t x1 = xyz1_[(i * n + j) * 2 + 0];
      scalar_t y1 = xyz1_[(i * n + j) * 2 + 1];
      int j2 = idx1[i * n + j];
      const scalar_t* xyz2_ = static_cast<const scalar_t*>(xyz2);
      scalar_t x2 = xyz2_[(i * m + j2) * 2 + 0];
      scalar_t y2 = xyz2_[(i * m + j2) * 2 + 1];
      const scalar_t* grad_dist1_ = static_cast<const scalar_t*>(grad_dist1);
      scalar_t g = grad_dist1_[i * n + j] * 2;
      scalar_t* grad_xyz1_ = static_cast<scalar_t*>(grad_xyz1);
      atomicAdd(&(grad_xyz1_[(i * n + j) * 2 + 0]), g * (x1 - x2));
      atomicAdd(&(grad_xyz1_[(i * n + j) * 2 + 1]), g * (y1 - y2));
      scalar_t* grad_xyz2_ = static_cast<scalar_t*>(grad_xyz2);
      atomicAdd(&(grad_xyz2_[(i * m + j2) * 2 + 0]), -(g * (x1 - x2)));
      atomicAdd(&(grad_xyz2_[(i * m + j2) * 2 + 1]), -(g * (y1 - y2)));
    }
  }
}

template <typename scalar_t>
__global__ void chamfer_distance_backward_cuda_kernel_diopi2(
    int b, int n, const void* xyz1_, int m, const void* xyz2_,
    const void* grad_dist1_, const int* idx1, void* grad_xyz1_,
    void* grad_xyz2_) {
  const scalar_t* xyz1 = static_cast<const scalar_t*>(xyz1_);
  const scalar_t* xyz2 = static_cast<const scalar_t*>(xyz2_);
  const scalar_t* grad_dist1 = static_cast<const scalar_t*>(grad_dist1_);
  scalar_t* grad_xyz1 = static_cast<scalar_t*>(grad_xyz1_);
  scalar_t* grad_xyz2 = static_cast<scalar_t*>(grad_xyz2_);

  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int j = threadIdx.x; j < n; j += blockDim.x * gridDim.y) {
      scalar_t x1 = xyz1[(i * n + j) * 2 + 0];
      scalar_t y1 = xyz1[(i * n + j) * 2 + 1];
      int j2 = idx1[i * n + j];
      scalar_t x2 = xyz2[(i * m + j2) * 2 + 0];
      scalar_t y2 = xyz2[(i * m + j2) * 2 + 1];
      scalar_t g = grad_dist1[i * n + j] * 2;
      atomicAdd(&(grad_xyz1[(i * n + j) * 2 + 0]), g * (x1 - x2));
      atomicAdd(&(grad_xyz1[(i * n + j) * 2 + 1]), g * (y1 - y2));
      atomicAdd(&(grad_xyz2[(i * m + j2) * 2 + 0]), -(g * (x1 - x2)));
      atomicAdd(&(grad_xyz2[(i * m + j2) * 2 + 1]), -(g * (y1 - y2)));
    }
  }
}

// DIOPI_API diopiError_t diopiChamferDistance(diopiContextHandle_t ctx, diopiConstTensorHandle_t xyz1_in, diopiConstTensorHandle_t xyz2_in, diopiTensorHandle_t dist1_out,
//                                             diopiTensorHandle_t dist2_out, diopiTensorHandle_t idx1_out, diopiTensorHandle_t idx2_out);

extern "C" diopiError_t diopiChamferDistance(diopiContextHandle_t ctx, diopiConstTensorHandle_t xyz1_in,
                     diopiConstTensorHandle_t xyz2_in, diopiTensorHandle_t dist1_out,
                     diopiTensorHandle_t dist2_out, diopiTensorHandle_t idx1_out,
                     diopiTensorHandle_t idx2_out) {
  auto xyz1 = impl::cuda::makeTensor(xyz1_in);
  auto xyz2 = impl::cuda::makeTensor(xyz2_in);
  auto dist1 = impl::cuda::makeTensor(dist1_out);
  auto dist2 = impl::cuda::makeTensor(dist2_out);
  auto idx1 = impl::cuda::makeTensor(idx1_out);
  auto idx2 = impl::cuda::makeTensor(idx2_out);
  int batch_size = xyz1.size(0);
  std::cout << "dkx fwd batch_size" << batch_size << std::endl;
  int n = xyz1.size(1);
  int m = xyz2.size(1);
  std::cout << "dkx fwd n" << n << std::endl;
  std::cout << "dkx fwd m" << m << std::endl;
  // here: wait for dipu ready
  // // at::cuda::CUDAGuard device_guard(xyz1.device());
  auto stream = impl::cuda::getStream(ctx);
  dispatch_float_types_and_half(chamfer_distance_forward_cuda_kernel_diopi, xyz1.dtype(), GET_BLOCKS(batch_size * n), THREADS_PER_BLOCK, stream,
                batch_size, n, xyz1.data(), m,
                xyz2.data(), dist1.data(),
                static_cast<int*>(idx1.data()));
  dispatch_float_types_and_half(chamfer_distance_forward_cuda_kernel_diopi, xyz1.dtype(), GET_BLOCKS(batch_size * m), THREADS_PER_BLOCK, stream,
                batch_size, m, xyz2.data(), n,
                xyz1.data(), dist2.data(),
                static_cast<int*>(idx2.data()));
  return diopiSuccess;
}

// extern "C" {
//     c10::DeviceType device2DeviceType(const diopiDevice_t device);
// }

// DIOPI_API diopiError_t diopiChamferDistanceBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t xyz1, diopiConstTensorHandle_t xyz2,
//                                             diopiConstTensorHandle_t idx1, diopiConstTensorHandle_t idx2, diopiConstTensorHandle_t grad_dist1, diopiConstTensorHandle_t grad_dist2,
//                                             diopiTensorHandle_t grad_xyz1, diopiTensorHandle_t grad_xyz2);

// extern c 和声明不一致。 DIOPI_API 这是一个 __attribute__((weak)) 的声明。
extern "C" diopiError_t diopiChamferDistanceBackward(
    diopiContextHandle_t ctx, diopiConstTensorHandle_t xyz1_in,
    diopiConstTensorHandle_t xyz2_in, diopiConstTensorHandle_t idx1_in,
    diopiConstTensorHandle_t idx2_in, diopiConstTensorHandle_t grad_dist1_in,
    diopiConstTensorHandle_t grad_dist2_in, diopiTensorHandle_t grad_xyz1_out,
    diopiTensorHandle_t grad_xyz2_out) {
  auto xyz1 = impl::cuda::makeTensor(xyz1_in);
  auto xyz2 = impl::cuda::makeTensor(xyz2_in);
  auto idx1 = impl::cuda::makeTensor(idx1_in);
  auto idx2 = impl::cuda::makeTensor(idx2_in);
  auto grad_dist1 = impl::cuda::makeTensor(grad_dist1_in);
  auto grad_dist2 = impl::cuda::makeTensor(grad_dist2_in);
  auto grad_xyz1 = impl::cuda::makeTensor(grad_xyz1_out);
  auto grad_xyz2 = impl::cuda::makeTensor(grad_xyz2_out);
  int batch_size = xyz1.size(0);
  std::cout << "dkx bwd batch_size" << batch_size << std::endl;
  int n = xyz1.size(1);
  int m = xyz2.size(1);
  std::cout << "dkx bwd n" << n << std::endl;
  std::cout << "dkx bwd m" << m << std::endl;
  // here: wait for dipu ready
  //// at::cuda::CUDAGuard device_guard(device2DeviceType(xyz1.device()));
  auto stream = impl::cuda::getStream(ctx);
  // dispatch_float_types_and_half(
  //               chamfer_distance_backward_cuda_kernel_diopi,
  //               xyz1.dtype(),
  //               GET_BLOCKS(batch_size * n),
  //               THREADS_PER_BLOCK / 2,
  //               stream,
  //               batch_size, m, xyz1.data(), n,
  //               xyz2.data(), grad_dist1.data(),
  //               static_cast<const int*>(idx1.data()),
  //               grad_xyz1.data(),
  //               grad_xyz2.data());
  dispatch_float_types_and_half(
                chamfer_distance_backward_cuda_kernel_diopi2,
                xyz1.dtype(),
                GET_BLOCKS(batch_size * n), THREADS_PER_BLOCK / 2, stream,
                batch_size, m, xyz1.data(), n,
                xyz2.data(), grad_dist1.data(),
                static_cast<const int*>(idx1.data()), grad_xyz1.data(),
                grad_xyz2.data());
  // dispatch_float_types_and_half(chamfer_distance_backward_cuda_kernel_diopi,
  //               xyz1.dtype(),
  //               GET_BLOCKS(batch_size * m),
  //               THREADS_PER_BLOCK / 2,
  //               stream,
  //               batch_size, n, xyz2.data(), m,
  //               xyz1.data(), grad_dist2.data(),
  //               static_cast<const int*>(idx2.data()),
  //               grad_xyz2.data(),
  //               grad_xyz1.data());
  dispatch_float_types_and_half(chamfer_distance_backward_cuda_kernel_diopi2,
                xyz1.dtype(),
                GET_BLOCKS(batch_size * m), THREADS_PER_BLOCK / 2, stream,
                batch_size, n, xyz2.data(), m,
                xyz1.data(), grad_dist2.data(),
                static_cast<const int*>(idx2.data()), grad_xyz2.data(),
                grad_xyz1.data());
  return diopiSuccess;
}

template <typename scalar_t>
__global__ void active_rotated_filter_forward_cuda_kernel_diopi(
    const int nthreads, const void* weight_data_, const int* indices_data,
    const int num_input_planes, const int num_output_planes,
    const int num_orientations, const int num_rotations, const int nEntry,
    void* output_data_) {
  const scalar_t* weight_data = static_cast<const scalar_t*>(weight_data_);
  scalar_t* output_data = static_cast<scalar_t*>(output_data_);
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int l = index % nEntry;
    int j = (index / nEntry) % num_input_planes;
    int i = index / nEntry / num_input_planes;
    int k;
    scalar_t val = *(weight_data + index);
    for (k = 0; k < num_rotations; k++) {
      int idx = (int)(*(indices_data + l * num_rotations + k)) - 1;
      scalar_t* target = output_data +
                         i * (num_rotations * num_input_planes * nEntry) +
                         k * (num_input_planes * nEntry) + j * (nEntry) + idx;
      *target = val;
    }
  }
}

template <typename scalar_t>
__global__ void active_rotated_filter_backward_cuda_kernel_diopi(
    const int nthreads, const void* gradWeight_data_,
    const int* indices_data, const int num_input_planes,
    const int num_output_planes, const int num_orientations,
    const int num_rotations, const int nEntry, void* weight_data_) {
  const scalar_t* gradWeight_data = static_cast<const scalar_t*>(gradWeight_data_);
  scalar_t* weight_data = static_cast<scalar_t*>(weight_data_);
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int l = index % nEntry;
    int j = (index / nEntry) % num_input_planes;
    int i = index / nEntry / num_input_planes;
    int k;
    scalar_t* val = weight_data + index;
    *val = 0;
    scalar_t tmp = 0;
    for (k = 0; k < num_rotations; k++) {
      int idx = (int)(*(indices_data + l * num_rotations + k)) - 1;
      scalar_t target =
          *(gradWeight_data + i * (num_rotations * num_input_planes * nEntry) +
            k * (num_input_planes * nEntry) + j * (nEntry) + idx);
      tmp = tmp + target;
    }
    *val = tmp;
  }
}

extern "C" diopiError_t diopiActiveRotatedFilter(diopiContextHandle_t ctx,
                                      diopiConstTensorHandle_t input_,
                                      diopiConstTensorHandle_t indices_,
                                      diopiTensorHandle_t output_) {
  auto input = impl::cuda::makeTensor(input_);
  auto indices = impl::cuda::makeTensor(indices_);
  auto output = impl::cuda::makeTensor(output_);

  int num_output_planes = input.size(0);
  int num_input_planes = input.size(1);
  int num_orientations = input.size(2);
  int kH = input.size(3);
  int kW = input.size(4);
  int num_rotations = indices.size(3);
  int nEntry = num_orientations * kH * kW;
  int output_size = input.numel();

  // // at::cuda::CUDAGuard device_guard(input.device());
  auto stream = impl::cuda::getStream(ctx);
  dispatch_float_types_and_half(
        active_rotated_filter_forward_cuda_kernel_diopi,
        input.dtype(),
        GET_BLOCKS(output_size), THREADS_PER_BLOCK, stream,
        output_size, input.data(),
        static_cast<const int*>(indices.data()), num_input_planes, num_output_planes,
        num_orientations, num_rotations, nEntry,
        output.data());
  return diopiSuccess;
}

extern "C" diopiError_t diopiActiveRotatedFilterBackward(
    diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out_,
    diopiConstTensorHandle_t indices_, diopiTensorHandle_t grad_in_) {
  auto grad_out = impl::cuda::makeTensor(grad_out_);
  auto indices = impl::cuda::makeTensor(indices_);
  auto grad_in = impl::cuda::makeTensor(grad_in_);

  int num_orientations = indices.size(0);
  int kH = indices.size(1);
  int kW = indices.size(2);
  int num_rotations = indices.size(3);
  int num_output_planes = grad_out.size(0) / num_rotations;
  int num_input_planes = grad_out.size(1) / num_orientations;
  int nEntry = num_orientations * kH * kW;
  int output_size = grad_in.numel();

  // // at::cuda::CUDAGuard device_guard(indices.device());
  auto stream = impl::cuda::getStream(ctx);
  dispatch_float_types_and_half(
        active_rotated_filter_backward_cuda_kernel_diopi,
        grad_out.scalar_type(),
        GET_BLOCKS(output_size), THREADS_PER_BLOCK, stream,
        output_size, grad_out.data(),
        static_cast<const int*>(indices.data()), num_input_planes, num_output_planes,
        num_orientations, num_rotations, nEntry,
        grad_in.data());
  return diopiSuccess;
}


template <typename T>
__global__ void assign_score_withk_forward_cuda_kernel_diopi(
    const int B, const int N0, const int N1, const int M, const int K,
    const int O, const int aggregate, const void* points_, const void* centers_,
    const void* scores_, const int64_t* knn_idx, void* output_) {
  const T* points = static_cast<const T*>(points_);
  const T* centers = static_cast<const T*>(centers_);
  const T* scores = static_cast<const T*>(scores_);
  T* output = static_cast<T*>(output_);
  // ----- parallel loop for B, N1, K and O ---------
  CUDA_1D_KERNEL_LOOP(i, B * O * N1 * K) {
    // ------- loop for M ----------
    const int b = (int)(i / (O * N1 * K));
    const int o = (int)(i % (O * N1 * K) / (N1 * K));
    const int n = (int)(i % (N1 * K) / K);
    const int k = (int)(i % K);
    const int cn = (int)knn_idx[b * K * N1 + n * K +
                                0];  // The first neighbor is the center point
    const int kn = (int)knn_idx[b * K * N1 + n * K + k];
    if (kn >= N0 ||
        kn < 0) {  // if index overflows, it is out of the neighborhood range
      return;
    }
    assert(b < B);
    assert(kn < N0);
    assert(cn < N0);
    assert(o < O);
    assert(n < N1);
    const int out_idx = b * N1 * O * K + o * N1 * K + n * K + k;
    T val = output[out_idx];
    for (int m = 0; m < M; m++) {
      val += points[b * N0 * M * O + kn * M * O + m * O + o] *
                 scores[b * N1 * K * M + n * K * M + k * M + m] -
             centers[b * N0 * M * O + cn * M * O + m * O + o] *
                 scores[b * N1 * K * M + n * K * M + k * M + m];
    }
    output[out_idx] = val;
  }
}

template <typename T>
__global__ void assign_score_withk_points_backward_cuda_kernel_diopi(
    const int B, const int N0, const int N, const int M, const int K,
    const int O, const int aggregate, const void* grad_out_, const void* scores_,
    const int64_t* knn_idx, void* grad_points_, void* grad_centers_) {
  const T* grad_out = static_cast<const T*>(grad_out_);
  const T* scores = static_cast<const T*>(scores_);
  T* grad_points = static_cast<T*>(grad_points_);
  T* grad_centers = static_cast<T*>(grad_centers_);
  // ----- parallel loop for B, M, O ---------
  CUDA_1D_KERNEL_LOOP(i, B * M * O) {
    int b = (int)(i / (M * O));
    int m = (int)(i % (M * O) / O);
    int o = (int)(i % O);

    // ----- loop for N,K ---------
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        int kn = knn_idx[b * N * K + n * K + k];
        int cn = knn_idx[b * N * K + n * K + 0];
        if (kn >= N0 || kn < 0) {  // if index overflows, it is out of the
                                   // neighborhood range
          continue;
        }
        atomicAdd(grad_points + b * N0 * M * O + kn * M * O + m * O + o,
                  scores[b * N * K * M + n * K * M + k * M + m] *
                      grad_out[b * O * N * K + o * N * K + n * K + k]);
        atomicAdd(grad_centers + b * N0 * M * O + cn * M * O + m * O + o,
                  -scores[b * N * K * M + n * K * M + k * M + m] *
                      grad_out[b * O * N * K + o * N * K + n * K + k]);
      }
    }
  }
}

template <typename T>
__global__ void assign_score_withk_scores_backward_cuda_kernel_diopi(
    const int B, const int N0, const int N, const int M, const int K,
    const int O, const int aggregate, const void* grad_out_, const void* points_,
    const void* centers_, const int64_t* knn_idx, void* grad_scores_) {
  const T* grad_out = static_cast<const T*>(grad_out_);
  const T* points = static_cast<const T*>(points_);
  const T* centers = static_cast<const T*>(centers_);
  T* grad_scores = static_cast<T*>(grad_scores_);
  // ----- parallel loop for B, N, K, M ---------
  CUDA_1D_KERNEL_LOOP(i, B * N * K * M) {
    const int b = (int)(i / (N * M * K));
    const int n = (int)(i % (N * M * K) / M / K);
    const int k = (int)(i % (M * K) / M);
    const int m = (int)(i % M);
    const int cn = knn_idx[b * N * K + n * K + 0];
    const int kn = knn_idx[b * N * K + n * K + k];
    if (kn >= N0 ||
        kn < 0) {  // if index overflows, it is out of the neighborhood range
      return;
    }

    // -------------- loop for O ------------------------
    const int out_idx = b * N * K * M + n * K * M + k * M + m;
    T val = grad_scores[out_idx];
    for (int o = 0; o < O; o++) {
      val += (points[b * N0 * M * O + kn * M * O + m * O + o] -
              centers[b * N0 * M * O + cn * M * O + m * O + o]) *
             grad_out[b * O * N * K + o * N * K + n * K + k];
    }
    grad_scores[out_idx] = val;
  }
}

diopiError_t diopiAssignScoreWithk(diopiContextHandle_t ctx,
                                   diopiConstTensorHandle_t points_,
                                   diopiConstTensorHandle_t centers_,
                                   diopiConstTensorHandle_t scores_,
                                   diopiConstTensorHandle_t knn_idx_,
                                   diopiTensorHandle_t output_, int64_t B,
                                   int64_t N0, int64_t N1, int64_t M, int64_t K,
                                   int64_t O, int64_t aggregate) {
  auto points = impl::cuda::makeTensor(points_);
  auto centers = impl::cuda::makeTensor(centers_);
  auto scores = impl::cuda::makeTensor(scores_);
  auto knn_idx = impl::cuda::makeTensor(knn_idx_);
  auto output = impl::cuda::makeTensor(output_);

  // // at::cuda::CUDAGuard device_guard(points.device());
  auto stream = impl::cuda::getStream(ctx);

  dim3 blocks(GET_BLOCKS(B * O * N1 * K, THREADS_PER_BLOCK));
  dim3 threads(THREADS_PER_BLOCK);

  dispatch_float_types_and_half(
                assign_score_withk_forward_cuda_kernel_diopi,
                points.scalar_type(),
                blocks, threads, stream,
                B, N0, N1, M, K, O, aggregate, points.data(),
                centers.data(), scores.data(),
                static_cast<const int64_t*>(knn_idx.data()), output.data());
  return diopiSuccess;
}


// 不加 extern C 可以么
diopiError_t diopiAssignScoreWithkBackward(
    diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out_,
    diopiConstTensorHandle_t points_, diopiConstTensorHandle_t centers_,
    diopiConstTensorHandle_t scores_, diopiConstTensorHandle_t knn_idx_,
    diopiTensorHandle_t grad_points_, diopiTensorHandle_t grad_centers_,
    diopiTensorHandle_t grad_scores_, int64_t B, int64_t N0, int64_t N1,
    int64_t M, int64_t K, int64_t O, int64_t aggregate) {
  auto grad_out = impl::cuda::makeTensor(grad_out_);
  auto points = impl::cuda::makeTensor(points_);
  auto centers = impl::cuda::makeTensor(centers_);
  auto scores = impl::cuda::makeTensor(scores_);
  auto knn_idx = impl::cuda::makeTensor(knn_idx_);
  auto grad_points = impl::cuda::makeTensor(grad_points_);
  auto grad_centers = impl::cuda::makeTensor(grad_centers_);
  auto grad_scores = impl::cuda::makeTensor(grad_scores_);

  // // at::cuda::CUDAGuard device_guard(grad_out.device());
  auto stream = impl::cuda::getStream(ctx);

  dim3 blocks1(GET_BLOCKS(B * M * O, THREADS_PER_BLOCK));
  dim3 threads1(THREADS_PER_BLOCK);
  dim3 blocks2(GET_BLOCKS(B * N1 * K * M, THREADS_PER_BLOCK));
  dim3 threads2(THREADS_PER_BLOCK);

  dispatch_float_types_and_half(
                assign_score_withk_points_backward_cuda_kernel_diopi,
                grad_out.scalar_type(),
                blocks1, threads1, stream,
                B, N0, N1, M, K, O, aggregate, grad_out.data(),
                scores.data(), static_cast<const int64_t*>(knn_idx.data()),
                grad_points.data(),
                grad_centers.data());

  dispatch_float_types_and_half(
                assign_score_withk_scores_backward_cuda_kernel_diopi,
                grad_out.scalar_type(),
                blocks2, threads2, stream,
                B, N0, N1, M, K, O, aggregate, grad_out.data(),
                points.data(), centers.data(),
                static_cast<const int64_t*>(knn_idx.data()), grad_scores.data());

  return diopiSuccess;
}

template <typename T>
__device__ __forceinline__ void load_bbox(const T* bbox, const int base, T& x1,
                                          T& y1, T& x2, T& y2) {
  x1 = bbox[base];
  y1 = bbox[base + 1];
  x2 = bbox[base + 2];
  y2 = bbox[base + 3];
}

template <>
__device__ __forceinline__ void load_bbox<float>(const float* bbox,
                                                 const int base, float& x1,
                                                 float& y1, float& x2,
                                                 float& y2) {
  const float4 bbox_offset = reinterpret_cast<const float4*>(bbox + base)[0];
  x1 = bbox_offset.x;
  y1 = bbox_offset.y;
  x2 = bbox_offset.z;
  y2 = bbox_offset.w;
}

template <typename T>
__global__ void bbox_overlaps_cuda_kernel(const void* bbox1_, const void* bbox2_,
                                          void* ious_, const int num_bbox1,
                                          const int num_bbox2, const int mode,
                                          const bool aligned,
                                          const int offset) {
  const T* bbox1 = static_cast<const T*>(bbox1_);
  const T* bbox2 = static_cast<const T*>(bbox2_);
  T* ious = static_cast<T*>(ious_);
  if (aligned) {
    CUDA_1D_KERNEL_LOOP(index, num_bbox1) {
      const int b1 = index;
      const int b2 = index;

      const int base1 = b1 << 2;  // b1 * 4
      T b1_x1, b1_y1, b1_x2, b1_y2;
      load_bbox<T>(bbox1, base1, b1_x1, b1_y1, b1_x2, b1_y2);
      const T b1_area = (b1_x2 - b1_x1 + offset) * (b1_y2 - b1_y1 + offset);

      const int base2 = b2 << 2;  // b2 * 4
      T b2_x1, b2_y1, b2_x2, b2_y2;
      load_bbox<T>(bbox2, base2, b2_x1, b2_y1, b2_x2, b2_y2);
      const T b2_area = (b2_x2 - b2_x1 + offset) * (b2_y2 - b2_y1 + offset);

      const T left = fmaxf(b1_x1, b2_x1), right = fminf(b1_x2, b2_x2);
      const T top = fmaxf(b1_y1, b2_y1), bottom = fminf(b1_y2, b2_y2);
      const T width = fmaxf(right - left + offset, 0.f);
      const T height = fmaxf(bottom - top + offset, 0.f);
      const T interS = width * height;

      const T baseS =
          fmaxf(mode == 0 ? b1_area + b2_area - interS : b1_area, T(offset));
      ious[index] = interS / baseS;
    }
  } else {
    CUDA_1D_KERNEL_LOOP(index, num_bbox1 * num_bbox2) {
      const int b1 = index / num_bbox2;
      const int b2 = index % num_bbox2;

      const int base1 = b1 << 2;  // b1 * 4
      T b1_x1, b1_y1, b1_x2, b1_y2;
      load_bbox<T>(bbox1, base1, b1_x1, b1_y1, b1_x2, b1_y2);
      const T b1_area = (b1_x2 - b1_x1 + offset) * (b1_y2 - b1_y1 + offset);

      const int base2 = b2 << 2;  // b2 * 4
      T b2_x1, b2_y1, b2_x2, b2_y2;
      load_bbox<T>(bbox2, base2, b2_x1, b2_y1, b2_x2, b2_y2);
      const T b2_area = (b2_x2 - b2_x1 + offset) * (b2_y2 - b2_y1 + offset);

      const T left = fmaxf(b1_x1, b2_x1), right = fminf(b1_x2, b2_x2);
      const T top = fmaxf(b1_y1, b2_y1), bottom = fminf(b1_y2, b2_y2);
      const T width = fmaxf(right - left + offset, 0.f);
      const T height = fmaxf(bottom - top + offset, 0.f);
      const T interS = width * height;

      const T baseS =
          fmaxf(mode == 0 ? b1_area + b2_area - interS : b1_area, T(offset));
      ious[index] = interS / baseS;
    }
  }
}

diopiError_t diopiBboxOverlaps(diopiContextHandle_t ctx,
                               diopiConstTensorHandle_t bboxes1_,
                               diopiConstTensorHandle_t bboxes2_,
                               diopiTensorHandle_t ious_, const int64_t mode,
                               const bool aligned, const int64_t offset) {
  auto bboxes1 = impl::cuda::makeTensor(bboxes1_);
  auto bboxes2 = impl::cuda::makeTensor(bboxes2_);
  auto ious = impl::cuda::makeTensor(ious_);
  int output_size = ious.numel();
  int num_bbox1 = bboxes1.size(0);
  int num_bbox2 = bboxes2.size(0);

  // // at::cuda::CUDAGuard device_guard(bboxes1.device());
  auto stream = impl::cuda::getStream(ctx);
  dispatch_float_types_and_half(
                bbox_overlaps_cuda_kernel,
                bboxes1.scalar_type()
                GET_BLOCKS(output_size), THREADS_PER_BLOCK, stream,
                bboxes1.data(), bboxes2.data(),
                ious.data(), num_bbox1, num_bbox2, mode, aligned,
                offset);
  return diopiSuccess;
}

#include <float.h>

// 这里是否会重复定义问题？如果和 mmcv 一起编译的话
enum BorderMode { Top = 0, Left = 1, Bottom = 2, Right = 3 };

/*** Forward ***/
template <typename T>
__global__ void border_align_forward_cuda_kernel(
    const int nthreads, const void* input_, const void* boxes_, void* output_,
    int* argmax_idx, const int channels, const int box_size, const int height,
    const int width, const int pool_size) {
  const T* input = static_cast<const T*>(input_);
  const T* boxes = static_cast<const T*>(boxes_);
  T* output = static_cast<T*>(output_);
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (batch_idx, c_idx, box_idx) is an element paralleled for computing
    // output, and `extreme_idx` is in range [0,3]
    int batch_idx, c_idx, box_idx, extreme_idx, maxidx, *offset_argmax_idx;
    const T *offset_box, *offset_input, *offset_box_x;
    T *offset_output, box_width, box_height, stride, x_stride, y_stride, x, y,
        val, maxval;

    extreme_idx = threadIdx.y;
    // shape (N, C, box_size, 4) for output
    batch_idx = index / channels / box_size;
    // shape (N, box_size, 4) for boxes
    box_idx = index % box_size + batch_idx * box_size;
    c_idx = (index / box_size) % channels;

    offset_box = boxes + box_idx * 4;
    box_width = *(offset_box + 2) - *offset_box;
    box_height = *(offset_box + 3) - *(offset_box + 1);
    offset_output = output + index * 4 + extreme_idx;
    offset_argmax_idx = argmax_idx + index * 4 + extreme_idx;
    // shape (N, 4C, h, w) for input.
    // [0,C) for top feature, [C,2C) for left feature,
    // [2C,3C) for bottom feature, [3C,4C) for right feature
    offset_input =
        input + (batch_idx * channels * 4 + extreme_idx * channels + c_idx) *
                    height * width;

    // extreme_idx in [0,1] -> offset_box_x indexed at x1
    // extreme_idx in [2,3] -> offset_box_x indexed at x2
    offset_box_x = offset_box + extreme_idx / 2 * 2;

    // (x1,y1) or (x2,y2) for (x,y)
    x = *offset_box_x;
    y = *(offset_box_x + 1);

    switch (extreme_idx) {
      // top
      case BorderMode::Top:
        stride = box_width / pool_size;
        x_stride = stride;
        y_stride = 0;
        break;
      // left
      case BorderMode::Left:
        stride = box_height / pool_size;
        x_stride = 0;
        y_stride = stride;
        break;
      // bottom
      case BorderMode::Bottom:
        stride = box_width / pool_size;
        x_stride = -stride;
        y_stride = 0;
        break;
      // right
      case BorderMode::Right:
        stride = box_height / pool_size;
        x_stride = 0;
        y_stride = -stride;
        break;
    }

    // initialize maxval and maxidx with the start position (e.g. (x1,y1) or
    // (x2,y2))
    maxval = bilinear_interpolate(offset_input, height, width, y, x, index);
    maxidx = 0;

    // do max_pool along the border
    for (int i = 1; i <= pool_size; i++) {
      x += x_stride;
      y += y_stride;
      val = bilinear_interpolate(offset_input, height, width, y, x, index);
      if (val > maxval) {
        maxval = val;
        maxidx = i;
      }
    }

    // update output and argmax_idx
    *offset_output = maxval;
    *offset_argmax_idx = maxidx;
  }
}

/*** Backward ***/
template <typename T>
__global__ void border_align_backward_cuda_kernel(
    const int nthreads, const void* grad_output_, const void* boxes_,
    const int* argmax_idx, void* grad_input_, const int channels,
    const int box_size, const int height, const int width,
    const int pool_size) {
  const T* grad_outpu = static_cast<const T*>(grad_output_);
  const T* boxes = static_cast<const T*>(boxes_);
  T* grad_input = static_cast<T*>(grad_input_);
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (batch_idx, c_idx, box_idx) is an element paralleled for computing
    // output, and `extreme_idx` is in range [0,3]
    int batch_idx, c_idx, box_idx, extreme_idx;
    const int* offset_argmax_idx;
    const T *offset_grad_output, *offset_box, *offset_box_x;
    T *offset_grad_input, box_width, box_height, stride, x_stride, y_stride, x,
        y;

    extreme_idx = threadIdx.y;
    batch_idx = index / channels / box_size;
    box_idx = index % box_size + batch_idx * box_size;
    c_idx = (index / box_size) % channels;

    offset_box = boxes + box_idx * 4;
    box_width = *(offset_box + 2) - *offset_box;
    box_height = *(offset_box + 3) - *(offset_box + 1);
    offset_grad_output = grad_output + index * 4 + extreme_idx;
    offset_argmax_idx = argmax_idx + index * 4 + extreme_idx;
    // [0,C) for top feature grad, [C,2C) for left feature grad,
    // [2C,3C) for bottom feature grad, [3C,4C) for right feature grad
    offset_grad_input = grad_input + (batch_idx * channels * 4 +
                                      extreme_idx * channels + c_idx) *
                                         height * width;

    // extreme_idx in [0,1] -> offset_box_x indexed at x1
    // extreme_idx in [2,3] -> offset_box_x indexed at x2
    offset_box_x = offset_box + extreme_idx / 2 * 2;

    switch (extreme_idx) {
      // top
      case BorderMode::Top:
        stride = box_width / pool_size;
        x_stride = stride;
        y_stride = 0;
        break;
      // left
      case BorderMode::Left:
        stride = box_height / pool_size;
        x_stride = 0;
        y_stride = stride;
        break;
      // bottom
      case BorderMode::Bottom:
        stride = box_width / pool_size;
        x_stride = -stride;
        y_stride = 0;
        break;
      // right
      case BorderMode::Right:
        stride = box_height / pool_size;
        x_stride = 0;
        y_stride = -stride;
        break;
    }

    // get position (x,y) which has maximum value during forward
    x = *offset_box_x;
    y = *(offset_box_x + 1);
    x += x_stride * (T)(*offset_argmax_idx);
    y += y_stride * (T)(*offset_argmax_idx);

    T w1, w2, w3, w4;
    int x_low, x_high, y_low, y_high;
    bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4, x_low,
                                  x_high, y_low, y_high, index);

    // update grad_output
    atomicAdd(offset_grad_input + y_low * width + x_low,
              *offset_grad_output * w1);
    atomicAdd(offset_grad_input + y_low * width + x_high,
              *offset_grad_output * w2);
    atomicAdd(offset_grad_input + y_high * width + x_low,
              *offset_grad_output * w3);
    atomicAdd(offset_grad_input + y_high * width + x_high,
              *offset_grad_output * w4);
  }
}

diopiError_t diopiBorderAlign(diopiContextHandle_t ctx, diopiConstTensorHandle_t input_,
                 diopiConstTensorHandle_t boxes_, diopiTensorHandle_t output_,
                 diopiTensorHandle_t argmax_idx_, const int64_t pool_size) {
  auto input = impl::cuda::makeTensor(input_);
  auto boxes = impl::cuda::makeTensor(boxes_);
  auto output = impl::cuda::makeTensor(output_);
  auto argmax_idx = impl::cuda::makeTensor(argmax_idx_);
  // shape assertion
  assert(input.ndimension() == 4);
  assert(boxes.ndimension() == 3);

  int batch_size = input.size(0);
  int feat_channels = input.size(1);
  int channels = feat_channels / 4;
  int height = input.size(2);
  int width = input.size(3);
  // shape [N, box_size, 4] for boxes. (x1, y1, x2, y2) format
  int box_size = boxes.size(1);
  // shape [N, channels, box_size, 4] for output
  int nthreads = batch_size * channels * box_size;

  // at::cuda::CUDAGuard device_guard(input.device());
  auto stream = impl::cuda::getStream(ctx);
  dim3 block(128, 4);
  dispatch_float_types_and_half(
                border_align_forward_cuda_kernel,
                input.scalar_type(),
                GET_BLOCKS(nthreads), block, stream,
                nthreads, input.data(),
                boxes.data(), output.data(),
                static_cast<int*>(argmax_idx.data()), channels, box_size, height, width,
                pool_size);
  return diopiSuccess;
}

diopiError_t diopiBorderAlignBackward(diopiContextHandle_t ctx,
                                      diopiConstTensorHandle_t grad_output_,
                                      diopiConstTensorHandle_t boxes_,
                                      diopiConstTensorHandle_t argmax_idx_,
                                      diopiTensorHandle_t grad_input_,
                                      const int64_t pool_size) {
  auto grad_output = impl::cuda::makeTensor(grad_output_);
  auto boxes = impl::cuda::makeTensor(boxes_);
  auto argmax_idx = impl::cuda::makeTensor(argmax_idx_);
  auto grad_input = impl::cuda::makeTensor(grad_input_);

  int batch_size = grad_input.size(0);
  int feat_channels = grad_input.size(1);
  int channels = feat_channels / 4;
  int height = grad_input.size(2);
  int width = grad_input.size(3);
  int box_size = boxes.size(1);
  int nthreads = batch_size * channels * box_size;

  // at::cuda::CUDAGuard device_guard(grad_output.device());
  auto stream = impl::cuda::getStream(ctx);
  dim3 block(128, 4);
  dispatch_float_types_and_half(
                border_align_backward_cuda_kernel,
                grad_output.scalar_type(),
                GET_BLOCKS(nthreads), block, stream,
                nthreads, grad_output.data(),
                boxes.data(), static_cast<const int*>(argmax_idx.data()),
                grad_input.data(), channels, box_size, height,
                width, pool_size);
  return diopiSuccess;
}

#define MAXN 100
#define NMAX 512

__device__ const double EPS = 1E-8;

__device__ inline int sig(double d) { return (d > EPS) - (d < -EPS); }

struct Point {
  double x, y;
  __device__ Point() {}
  __device__ Point(double x, double y) : x(x), y(y) {}
};

__device__ inline bool point_same(Point& a, Point& b) {
  return sig(a.x - b.x) == 0 && sig(a.y - b.y) == 0;
}

__device__ inline void swap1(Point* a, Point* b) {
  Point temp;
  temp.x = a->x;
  temp.y = a->y;

  a->x = b->x;
  a->y = b->y;

  b->x = temp.x;
  b->y = temp.y;
}

__device__ inline void reverse1(Point* a, const int n) {
  for (int i = 0; i < (n - 1) / 2.0; i++) {
    Point* j = &(a[i]);
    Point* k = &(a[n - 1 - i]);
    swap1(j, k);
  }
}

__device__ inline double cross(Point o, Point a, Point b) {
  return (a.x - o.x) * (b.y - o.y) - (b.x - o.x) * (a.y - o.y);
}

__device__ inline double dis(Point a, Point b) {
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}
__device__ inline double area(Point* ps, int n) {
  ps[n] = ps[0];
  double res = 0;
  for (int i = 0; i < n; i++) {
    res += ps[i].x * ps[i + 1].y - ps[i].y * ps[i + 1].x;
  }
  return res / 2.0;
}
__device__ inline double polygon_area_grad(Point* ps, int n,
                                           int* polygon_to_pred_index,
                                           int n_pred, double* grad_C) {
  ps[n] = ps[0];
  double partion_grad[4 * 30 + 2];
  double res = 0;
  for (int i = 0; i < n; i++) {
    res += ps[i].x * ps[i + 1].y - ps[i].y * ps[i + 1].x;
    partion_grad[i * 4 + 2] = ps[i + 1].y;
    partion_grad[i * 4 + 3] = -ps[i + 1].x;
    if (i != n - 1) {
      partion_grad[i * 4 + 4] = -ps[i].y;
      partion_grad[i * 4 + 5] = ps[i].x;
    } else {
      partion_grad[0] = -ps[i].y;
      partion_grad[1] = ps[i].x;
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n_pred; j++) {
      if (i == polygon_to_pred_index[j]) {
        grad_C[2 * polygon_to_pred_index[j + n_pred]] =
            (partion_grad[i * 4] + partion_grad[i * 4 + 2]) / 2;
        break;
      }
    }
    for (int j = 0; j < n_pred; j++) {
      if (i == polygon_to_pred_index[j]) {
        grad_C[2 * polygon_to_pred_index[j + n_pred] + 1] =
            (partion_grad[i * 4 + 1] + partion_grad[i * 4 + 1 + 2]) / 2;
        break;
      }
    }
  }

  return res / 2.0;
}

__device__ inline int lineCross(Point a, Point b, Point c, Point d, Point& p,
                                double* cut_grad, int m, int n, int i) {
  double s1, s2;
  double s2_s1_2;
  double ds1_dxc, ds1_dyc, ds2_dxd, ds2_dyd;
  double dxp_dxc, dxp_dyc, dxp_dxd, dxp_dyd, dyp_dxc, dyp_dyc, dyp_dxd, dyp_dyd;
  s1 = cross(a, b, c);
  s2 = cross(a, b, d);

  ds1_dxc = -(b.y - a.y);
  ds1_dyc = b.x - a.x;
  ds2_dxd = ds1_dxc;
  ds2_dyd = ds1_dyc;
  s2_s1_2 = (s2 - s1) * (s2 - s1);

  if (sig(s1) == 0 && sig(s2) == 0) return 2;
  if (sig(s2 - s1) == 0) return 0;

  dxp_dxc =
      ((s2 - d.x * ds1_dxc) * (s2 - s1) - (c.x * s2 - d.x * s1) * (-ds1_dxc)) /
      (s2_s1_2);
  dxp_dyc =
      ((0 - d.x * ds1_dyc) * (s2 - s1) - (c.x * s2 - d.x * s1) * (-ds1_dyc)) /
      (s2_s1_2);
  dxp_dxd =
      ((c.x * ds2_dxd - s1) * (s2 - s1) - (c.x * s2 - d.x * s1) * (ds2_dxd)) /
      (s2_s1_2);
  dxp_dyd =
      ((c.x * ds2_dyd - 0) * (s2 - s1) - (c.x * s2 - d.x * s1) * (ds2_dyd)) /
      (s2_s1_2);

  dyp_dxc =
      ((0 - d.y * ds1_dxc) * (s2 - s1) - (c.y * s2 - d.y * s1) * (-ds1_dxc)) /
      (s2_s1_2);
  dyp_dyc =
      ((s2 - d.y * ds1_dyc) * (s2 - s1) - (c.y * s2 - d.y * s1) * (-ds1_dyc)) /
      (s2_s1_2);
  dyp_dxd =
      ((c.y * ds2_dxd - 0) * (s2 - s1) - (c.y * s2 - d.y * s1) * (ds2_dxd)) /
      (s2_s1_2);
  dyp_dyd =
      ((c.y * ds2_dyd - s1) * (s2 - s1) - (c.y * s2 - d.y * s1) * (ds2_dyd)) /
      (s2_s1_2);

  p.x = (c.x * s2 - d.x * s1) / (s2 - s1);
  p.y = (c.y * s2 - d.y * s1) / (s2 - s1);
  if (i == n - 1) {
    cut_grad[4 * n * m + 4 * i] = dxp_dxc;  // + dyp_dxc;
    cut_grad[4 * n * m + 4 * i + 1] = dyp_dxc;
    cut_grad[4 * n * m + 4 * i + 2] = dxp_dyc;  // + dyp_dyc;
    cut_grad[4 * n * m + 4 * i + 3] = dyp_dyc;
    cut_grad[4 * n * m + 0] = dxp_dxd;  // + dyp_dxd;
    cut_grad[4 * n * m + 1] = dyp_dxd;
    cut_grad[4 * n * m + 2] = dxp_dyd;  // + dyp_dyd;
    cut_grad[4 * n * m + 3] = dyp_dyd;
  } else {
    cut_grad[4 * n * m + 4 * i] = dxp_dxc;  // + dyp_dxc;
    cut_grad[4 * n * m + 4 * i + 1] = dyp_dxc;
    cut_grad[4 * n * m + 4 * i + 2] = dxp_dyc;  // + dyp_dyc;
    cut_grad[4 * n * m + 4 * i + 3] = dyp_dyc;
    cut_grad[4 * n * m + 4 * (i + 1)] = dxp_dxd;  // + dyp_dxd;
    cut_grad[4 * n * m + 4 * (i + 1) + 1] = dyp_dxd;
    cut_grad[4 * n * m + 4 * (i + 1) + 2] = dxp_dyd;  // + dyp_dyd;
    cut_grad[4 * n * m + 4 * (i + 1) + 3] = dyp_dyd;
  }

  return 1;
}
__device__ inline void polygon_cut(Point* p, int& n, Point a, Point b,
                                   double* cut_grad) {
  Point pp[MAXN];
  double ccur_grad[MAXN] = {};
  int m = 0;
  p[n] = p[0];
  int k = n;
  for (int i = 0; i < n; i++) {
    if (sig(cross(a, b, p[i])) > 0) {
      pp[m] = p[i];
      ccur_grad[4 * n * m + 4 * i] = 1.0;
      ccur_grad[4 * n * m + 4 * i + 3] = 1.0;
      m++;
    }
    if (sig(cross(a, b, p[i])) != sig(cross(a, b, p[i + 1]))) {
      lineCross(a, b, p[i], p[i + 1], pp[m], ccur_grad, m, n, i);
      m++;
    }
  }

  n = 0;
  for (int i = 0; i < m; i++) {
    if (!i || !(point_same(pp[i], pp[i - 1]))) {
      p[n] = pp[i];
      for (int j = 0; j < 4 * k; j++) {
        cut_grad[4 * k * n + j] = ccur_grad[4 * k * i + j];
      }
      n++;
    }
  }

  while (n > 1 && point_same(p[n - 1], p[0])) n--;
}

__device__ inline double intersectArea(Point a, Point b, Point c, Point d,
                                       double* grad_AB, int order,
                                       int convex_n) {
  Point o(0, 0);
  int res_flag = 0;
  int s1 = sig(cross(o, a, b));
  int s2 = sig(cross(o, c, d));
  if (s1 == 0 || s2 == 0) return 0.0;
  if (s1 == -1) {
    Point* i = &a;
    Point* j = &b;
    swap1(i, j);
    res_flag = 1;
  }
  if (s2 == -1) {
    Point* i = &c;
    Point* j = &d;
    swap1(i, j);
  }
  Point p[10] = {o, a, b};
  int n = 3, n0 = 3, n1, n2, n3;
  double cut_grad1[MAXN] = {};
  double cut_grad2[MAXN] = {};
  double cut_grad3[MAXN] = {};
  double p1_p_grad[10][10] = {};
  double p2_p1_grad[10][10] = {};
  double p3_p2_grad[10][10] = {};

  double p3_p1_grad[10][10] = {};
  double p3_p_grad[10][10] = {};

  // 1
  polygon_cut(p, n, o, c, cut_grad1);
  n1 = n;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 4 * n0; j++) {
      if (!(j % 2)) {
        p1_p_grad[2 * i][j / 2] = cut_grad1[4 * n0 * i + j];
      } else {
        p1_p_grad[2 * i + 1][j / 2] = cut_grad1[4 * n0 * i + j];
      }
    }
  }

  // 2
  polygon_cut(p, n, c, d, cut_grad2);
  n2 = n;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 4 * n1; j++) {
      if (!(j % 2)) {
        p2_p1_grad[2 * i][j / 2] = cut_grad2[4 * n1 * i + j];
      } else {
        p2_p1_grad[2 * i + 1][j / 2] = cut_grad2[4 * n1 * i + j];
      }
    }
  }
  // 3
  polygon_cut(p, n, d, o, cut_grad3);
  n3 = n;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 4 * n2; j++) {
      if (!(j % 2)) {
        p3_p2_grad[2 * i][j / 2] = cut_grad3[4 * n2 * i + j];
      } else {
        p3_p2_grad[2 * i + 1][j / 2] = cut_grad3[4 * n2 * i + j];
      }
    }
  }

  // mul
  //  p3_p2(n3 * n2) * p2_p1(n2 * n1) = p3_p1 (n3 * n1)
  for (int i = 0; i < 2 * n3; i++) {
    for (int j = 0; j < 2 * n1; j++) {
      double sum = 0.0;
      for (int m = 0; m < 2 * n2; m++) {
        sum = sum + p3_p2_grad[i][m] * p2_p1_grad[m][j];
      }
      p3_p1_grad[i][j] = sum;
    }
  }

  // p3_p1 (n3 * n1) * p1_p (n1 * n0) = p3_p (n3 * n0)
  for (int i = 0; i < 2 * n3; i++) {
    for (int j = 0; j < 2 * n0; j++) {
      double sum = 0.0;
      for (int m = 0; m < 2 * n1; m++) {
        sum = sum + p3_p1_grad[i][m] * p1_p_grad[m][j];
      }
      p3_p_grad[i][j] = sum;
    }
  }

  // calculate S_grad
  int polygon_index_box_index[20];
  double grad_polygon[20];
  double S_grad[6];

  for (int i = 0; i < n3; i++) {
    polygon_index_box_index[i] = i;
    polygon_index_box_index[i + n3] = i;
  }

  double res =
      polygon_area_grad(p, n3, polygon_index_box_index, n3, grad_polygon);

  if (s1 * s2 == -1) {
    for (int j = 0; j < 2 * 3; j++) {
      double sum = 0.0;
      for (int m = 0; m < 2 * n3; m++) {
        sum = sum - grad_polygon[m] * p3_p_grad[m][j];
      }
      S_grad[j] = sum;
    }

    if (order != convex_n - 1) {
      if (res_flag) {
        grad_AB[2 * order] += S_grad[4];
        grad_AB[2 * order + 1] += S_grad[5];
        grad_AB[2 * order + 2] += S_grad[2];
        grad_AB[2 * order + 3] += S_grad[3];

      } else {
        grad_AB[2 * order] += S_grad[2];
        grad_AB[2 * order + 1] += S_grad[3];
        grad_AB[2 * order + 2] += S_grad[4];
        grad_AB[2 * order + 3] += S_grad[5];
      }
    } else {
      if (res_flag) {
        grad_AB[2 * order] += S_grad[4];
        grad_AB[2 * order + 1] += S_grad[5];
        grad_AB[0] += S_grad[2];
        grad_AB[1] += S_grad[3];

      } else {
        grad_AB[2 * order] += S_grad[2];
        grad_AB[2 * order + 1] += S_grad[3];
        grad_AB[0] += S_grad[4];
        grad_AB[1] += S_grad[5];
      }
    }
    res = -res;
  } else {
    for (int j = 0; j < 2 * 3; j++) {
      double sum = 0.0;
      for (int m = 0; m < 2 * n3; m++) {
        sum = sum + grad_polygon[m] * p3_p_grad[m][j];
      }
      S_grad[j] = sum;
    }

    if (order != convex_n - 1) {
      if (res_flag) {
        grad_AB[2 * order] += S_grad[4];
        grad_AB[2 * order + 1] += S_grad[5];
        grad_AB[2 * order + 2] += S_grad[2];
        grad_AB[2 * order + 3] += S_grad[3];
      } else {
        grad_AB[2 * order] += S_grad[2];
        grad_AB[2 * order + 1] += S_grad[3];
        grad_AB[2 * order + 2] += S_grad[4];
        grad_AB[2 * order + 3] += S_grad[5];
      }
    } else {
      if (res_flag) {
        grad_AB[2 * order] += S_grad[4];
        grad_AB[2 * order + 1] += S_grad[5];
        grad_AB[0] += S_grad[2];
        grad_AB[1] += S_grad[3];
      } else {
        grad_AB[2 * order] += S_grad[2];
        grad_AB[2 * order + 1] += S_grad[3];
        grad_AB[0] += S_grad[4];
        grad_AB[1] += S_grad[5];
      }
    }
  }
  return res;
}

__device__ inline double intersectAreaO(Point* ps1, int n1, Point* ps2, int n2,
                                        double* grad_AB) {
  if (area(ps1, n1) < 0) reverse1(ps1, n1);
  if (area(ps2, n2) < 0) reverse1(ps2, n2);
  ps1[n1] = ps1[0];
  ps2[n2] = ps2[0];
  double res = 0;
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      res +=
          intersectArea(ps1[i], ps1[i + 1], ps2[j], ps2[j + 1], grad_AB, i, n1);
    }
  }
  return res;
}

__device__ inline void Jarvis(Point* in_poly, int& n_poly) {
  Point p_max, p_k;
  int max_index, k_index;
  int Stack[NMAX] = {}, top1, top2;
  double sign;
  Point right_point[10], left_point[10];

  for (int i = 0; i < n_poly; i++) {
    if (in_poly[i].y < in_poly[0].y ||
        in_poly[i].y == in_poly[0].y && in_poly[i].x < in_poly[0].x) {
      Point* j = &(in_poly[0]);
      Point* k = &(in_poly[i]);
      swap1(j, k);
    }
    if (i == 0) {
      p_max = in_poly[0];
      max_index = 0;
    }
    if (in_poly[i].y > p_max.y ||
        in_poly[i].y == p_max.y && in_poly[i].x > p_max.x) {
      p_max = in_poly[i];
      max_index = i;
    }
  }

  if (max_index == 0) {
    max_index = 1;
    p_max = in_poly[max_index];
  }

  k_index = 0, Stack[0] = 0, top1 = 0;
  while (k_index != max_index) {
    p_k = p_max;
    k_index = max_index;
    for (int i = 1; i < n_poly; i++) {
      sign = cross(in_poly[Stack[top1]], in_poly[i], p_k);
      if ((sign > 0) || ((sign == 0) && (dis(in_poly[Stack[top1]], in_poly[i]) >
                                         dis(in_poly[Stack[top1]], p_k)))) {
        p_k = in_poly[i];
        k_index = i;
      }
    }
    top1++;
    Stack[top1] = k_index;
  }
  for (int i = 0; i <= top1; i++) right_point[i] = in_poly[Stack[i]];

  k_index = 0, Stack[0] = 0, top2 = 0;

  while (k_index != max_index) {
    p_k = p_max;
    k_index = max_index;
    for (int i = 1; i < n_poly; i++) {
      sign = cross(in_poly[Stack[top2]], in_poly[i], p_k);
      if ((sign < 0) || (sign == 0) && (dis(in_poly[Stack[top2]], in_poly[i]) >
                                        dis(in_poly[Stack[top2]], p_k))) {
        p_k = in_poly[i];
        k_index = i;
      }
    }
    top2++;
    Stack[top2] = k_index;
  }
  for (int i = top2 - 1; i >= 0; i--) left_point[i] = in_poly[Stack[i]];

  for (int i = 0; i < top1 + top2; i++) {
    if (i <= top1) {
      in_poly[i] = right_point[i];
    } else {
      in_poly[i] = left_point[top2 - (i - top1)];
    }
  }
  n_poly = top1 + top2;
}

__device__ inline double intersectAreaPoly(Point* ps1, int n1, Point* ps2,
                                           int n2, double* grad_C) {
  Point polygon[MAXN];
  int n = n1 + n2, n_poly = 0;
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n - n1; j++) {
      if (point_same(ps1[i], ps2[j])) {
        for (int k = j; k < n - n1 - 1; k++) {
          ps2[k] = ps2[k + 1];
        }
        n2--;
        break;
      }
    }
  }
  n_poly = n1 + n2;
  for (int i = 0; i < n_poly; i++) {
    if (i < n1) {
      polygon[i] = ps1[i];
    } else {
      polygon[i] = ps2[i - n1];
    }
  }

  Jarvis(polygon, n_poly);

  int polygon_to_pred_index[18] = {-1, -1, -1, -1, -1, -1, -1, -1, -1,
                                   -1, -1, -1, -1, -1, -1, -1, -1, -1};
  int n_pred = 0;
  for (int i = 0; i < n_poly; i++) {
    for (int j = 0; j < n1; j++) {
      if (polygon[i].x == ps1[j].x && polygon[i].y == ps1[j].y) {
        polygon_to_pred_index[n_pred] = i;
        polygon_to_pred_index[n_pred + n1] = j;
        n_pred += 1;
        break;
      }
    }
  }
  if (n_pred == 0) {
    double polygon_area = fabs(area(polygon, n_poly));
    for (int i = 0; i < 18; i++) {
      grad_C[i] = 0.0;
    }
    return polygon_area;
  } else {
    double polygon_area =
        polygon_area_grad(polygon, n_poly, polygon_to_pred_index, n1, grad_C);
    if (polygon_area < 0) {
      for (int i = 0; i < 18; i++) {
        grad_C[i] = -grad_C[i];
      }
    }
    return fabs(polygon_area);
  }
}

// convex_find and get the polygon_index_box_index
__device__ inline void Jarvis_and_index(Point* in_poly, int& n_poly,
                                        int* points_to_convex_ind) {
  int n_input = n_poly;
  Point input_poly[20];
  for (int i = 0; i < n_input; i++) {
    input_poly[i].x = in_poly[i].x;
    input_poly[i].y = in_poly[i].y;
  }
  Point p_max, p_k;
  int max_index, k_index;
  int Stack[20], top1, top2;
  double sign;
  Point right_point[10], left_point[10];

  for (int i = 0; i < n_poly; i++) {
    if (in_poly[i].y < in_poly[0].y ||
        in_poly[i].y == in_poly[0].y && in_poly[i].x < in_poly[0].x) {
      Point* j = &(in_poly[0]);
      Point* k = &(in_poly[i]);
      swap1(j, k);
    }
    if (i == 0) {
      p_max = in_poly[0];
      max_index = 0;
    }
    if (in_poly[i].y > p_max.y ||
        in_poly[i].y == p_max.y && in_poly[i].x > p_max.x) {
      p_max = in_poly[i];
      max_index = i;
    }
  }
  if (max_index == 0) {
    max_index = 1;
    p_max = in_poly[max_index];
  }

  k_index = 0, Stack[0] = 0, top1 = 0;
  while (k_index != max_index) {
    p_k = p_max;
    k_index = max_index;
    for (int i = 1; i < n_poly; i++) {
      sign = cross(in_poly[Stack[top1]], in_poly[i], p_k);
      if ((sign > 0) || ((sign == 0) && (dis(in_poly[Stack[top1]], in_poly[i]) >
                                         dis(in_poly[Stack[top1]], p_k)))) {
        p_k = in_poly[i];
        k_index = i;
      }
    }
    top1++;
    Stack[top1] = k_index;
  }
  for (int i = 0; i <= top1; i++) {
    right_point[i] = in_poly[Stack[i]];
  }

  k_index = 0, Stack[0] = 0, top2 = 0;

  while (k_index != max_index) {
    p_k = p_max;
    k_index = max_index;
    for (int i = 1; i < n_poly; i++) {
      sign = cross(in_poly[Stack[top2]], in_poly[i], p_k);
      if ((sign < 0) || (sign == 0) && (dis(in_poly[Stack[top2]], in_poly[i]) >
                                        dis(in_poly[Stack[top2]], p_k))) {
        p_k = in_poly[i];
        k_index = i;
      }
    }
    top2++;
    Stack[top2] = k_index;
  }

  for (int i = top2 - 1; i >= 0; i--) {
    left_point[i] = in_poly[Stack[i]];
  }

  for (int i = 0; i < top1 + top2; i++) {
    if (i <= top1) {
      in_poly[i] = right_point[i];
    } else {
      in_poly[i] = left_point[top2 - (i - top1)];
    }
  }
  n_poly = top1 + top2;
  for (int i = 0; i < n_poly; i++) {
    for (int j = 0; j < n_input; j++) {
      if (point_same(in_poly[i], input_poly[j])) {
        points_to_convex_ind[i] = j;
        break;
      }
    }
  }
}

template <typename T>
__device__ inline float devrIoU(T const* const p, T const* const q,
                                T* point_grad, const int idx) {
  Point ps1[MAXN], ps2[MAXN];

  Point convex[MAXN];
  for (int i = 0; i < 9; i++) {
    convex[i].x = (double)p[i * 2];
    convex[i].y = (double)p[i * 2 + 1];
  }
  int n_convex = 9;
  int points_to_convex_ind[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
  Jarvis_and_index(convex, n_convex, points_to_convex_ind);

  int n1 = n_convex;
  int n2 = 4;

  for (int i = 0; i < n1; i++) {
    ps1[i].x = (double)convex[i].x;
    ps1[i].y = (double)convex[i].y;
  }

  for (int i = 0; i < n2; i++) {
    ps2[i].x = (double)q[i * 2];
    ps2[i].y = (double)q[i * 2 + 1];
  }

  int polygon_index_box_index[18];
  for (int i = 0; i < n1; i++) {
    polygon_index_box_index[i] = i;
    polygon_index_box_index[i + n1] = i;
  }

  double grad_A[18] = {};
  double grad_AB[18] = {};
  double grad_C[18] = {};

  double inter_area = intersectAreaO(ps1, n1, ps2, n2, grad_AB);
  double S_pred =
      polygon_area_grad(ps1, n1, polygon_index_box_index, n1, grad_A);
  if (S_pred < 0) {
    for (int i = 0; i < n_convex * 2; i++) {
      grad_A[i] = -grad_A[i];
    }
  }
  double union_area = fabs(S_pred) + fabs(area(ps2, n2)) - inter_area;

  double iou = inter_area / union_area;
  double polygon_area = intersectAreaPoly(ps1, n1, ps2, n2, grad_C);

  //    printf("%d:live\n", idx);
  double rot_giou = iou - (polygon_area - union_area) / polygon_area;

  float grad_point_temp[18] = {};

  for (int i = 0; i < n_convex; i++) {
    int grad_point = points_to_convex_ind[i];
    grad_point_temp[2 * grad_point] =
        (float)((union_area + inter_area) / (union_area * union_area) *
                    grad_AB[2 * i] -
                iou / union_area * grad_A[2 * i] -
                1 / polygon_area * (grad_AB[2 * i] - grad_A[2 * i]) -
                (union_area) / polygon_area / polygon_area * grad_C[2 * i]);
    grad_point_temp[2 * grad_point + 1] =
        (float)((union_area + inter_area) / (union_area * union_area) *
                    grad_AB[2 * i + 1] -
                iou / union_area * grad_A[2 * i + 1] -
                1 / polygon_area * (grad_AB[2 * i + 1] - grad_A[2 * i + 1]) -
                (union_area) / polygon_area / polygon_area * grad_C[2 * i + 1]);
  }

  for (int i = 0; i < 9; i++) {
    point_grad[2 * i] = grad_point_temp[2 * i];
    point_grad[2 * i + 1] = grad_point_temp[2 * i + 1];
  }
  return (float)rot_giou;
}

template <typename T>
__global__ void convex_giou_cuda_kernel(const int ex_n_boxes,
                                        const int gt_n_boxes, const void* ex_boxes_,
                                        const void* gt_boxes_, void* point_grad_) {
  const T* ex_boxes = static_cast<const T*>(ex_boxes_);
  const T* gt_boxes = static_cast<const T*>(gt_boxes_);
  T* point_grad = static_cast<T*>(point_grad_);
  CUDA_1D_KERNEL_LOOP(index, ex_n_boxes) {
    const T* cur_box = ex_boxes + index * 18;
    const T* cur_gt_box = gt_boxes + index * 8;
    T* cur_grad = point_grad + index * 19;
    T giou = devrIoU(cur_box, cur_gt_box, cur_grad, threadIdx.x);
    cur_grad[18] = giou;
  }
}

__device__ inline int lineCross(Point a, Point b, Point c, Point d, Point& p) {
  double s1, s2;
  s1 = cross(a, b, c);
  s2 = cross(a, b, d);
  if (sig(s1) == 0 && sig(s2) == 0) return 2;
  if (sig(s2 - s1) == 0) return 0;
  p.x = (c.x * s2 - d.x * s1) / (s2 - s1);
  p.y = (c.y * s2 - d.y * s1) / (s2 - s1);
  return 1;
}

__device__ inline void polygon_cut(Point* p, int& n, Point a, Point b) {
  Point pp[MAXN];
  int m = 0;
  p[n] = p[0];
  for (int i = 0; i < n; i++) {
    if (sig(cross(a, b, p[i])) > 0) {
      pp[m] = p[i];
      m++;
    }
    if (sig(cross(a, b, p[i])) != sig(cross(a, b, p[i + 1]))) {
      lineCross(a, b, p[i], p[i + 1], pp[m]);
      m++;
    }
  }
  n = 0;
  for (int i = 0; i < m; i++) {
    if (!i || !(point_same(pp[i], pp[i - 1]))) {
      p[n] = pp[i];
      n++;
    }
  }

  while (n > 1 && point_same(p[n - 1], p[0])) n--;
}

__device__ inline double intersectArea(Point a, Point b, Point c, Point d) {
  Point o(0, 0);
  int s1 = sig(cross(o, a, b));
  int s2 = sig(cross(o, c, d));
  if (s1 == 0 || s2 == 0) return 0.0;
  if (s1 == -1) {
    Point* i = &a;
    Point* j = &b;
    swap1(i, j);
  }
  if (s2 == -1) {
    Point* i = &c;
    Point* j = &d;
    swap1(i, j);
  }
  Point p[10] = {o, a, b};
  int n = 3;

  polygon_cut(p, n, o, c);
  polygon_cut(p, n, c, d);
  polygon_cut(p, n, d, o);
  double res = area(p, n);
  if (s1 * s2 == -1) res = -res;
  return res;
}
__device__ inline double intersectAreaO(Point* ps1, int n1, Point* ps2,
                                        int n2) {
  if (area(ps1, n1) < 0) reverse1(ps1, n1);
  if (area(ps2, n2) < 0) reverse1(ps2, n2);
  ps1[n1] = ps1[0];
  ps2[n2] = ps2[0];
  double res = 0;
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      res += intersectArea(ps1[i], ps1[i + 1], ps2[j], ps2[j + 1]);
    }
  }
  return res;
}

template <typename T>
__device__ inline float devrIoU(T const* const p, T const* const q) {
  Point ps1[MAXN], ps2[MAXN];
  Point convex[MAXN];
  for (int i = 0; i < 9; i++) {
    convex[i].x = (double)p[i * 2];
    convex[i].y = (double)p[i * 2 + 1];
  }
  int n_convex = 9;
  int points_to_convex_ind[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
  Jarvis_and_index(convex, n_convex, points_to_convex_ind);
  int n1 = n_convex;
  for (int i = 0; i < n1; i++) {
    ps1[i].x = (double)convex[i].x;
    ps1[i].y = (double)convex[i].y;
  }
  int n2 = 4;
  for (int i = 0; i < n2; i++) {
    ps2[i].x = (double)q[i * 2];
    ps2[i].y = (double)q[i * 2 + 1];
  }
  double inter_area = intersectAreaO(ps1, n1, ps2, n2);
  double S_pred = area(ps1, n1);
  double union_area = fabs(S_pred) + fabs(area(ps2, n2)) - inter_area;
  double iou = inter_area / union_area;
  return (float)iou;
}

template <typename T>
__global__ void convex_iou_cuda_kernel(const int ex_n_boxes,
                                       const int gt_n_boxes, const void* ex_boxes_,
                                       const void* gt_boxes_, void* iou_) {
  const T* ex_boxes = static_cast<const T*>(ex_boxes_);
  const T* gt_boxes = static_cast<const T*>(gt_boxes_);
  T* iou = static_cast<T*>(iou_);
  CUDA_1D_KERNEL_LOOP(index, ex_n_boxes) {
    const T* cur_box = ex_boxes + index * 18;
    for (int i = 0; i < gt_n_boxes; i++) {
      iou[index * gt_n_boxes + i] = devrIoU(cur_box, gt_boxes + i * 8);
    }
  }
}

diopiError_t diopiConvexIou(diopiContextHandle_t ctx,
                            diopiConstTensorHandle_t pointsets_,
                            diopiConstTensorHandle_t polygons_,
                            diopiTensorHandle_t ious_) {
  auto pointsets = impl::cuda::makeTensor(pointsets_);
  auto polygons = impl::cuda::makeTensor(polygons_);
  auto ious = impl::cuda::makeTensor(ious_);

  int output_size = ious.numel();
  int num_pointsets = pointsets.size(0);
  int num_polygons = polygons.size(0);

  // at::cuda::CUDAGuard device_guard(pointsets.device());
  auto stream = impl::cuda::getStream(ctx);
  dispatch_float_types_and_half(
                convex_iou_cuda_kernel,
                pointsets.scalar_type(),
                GET_BLOCKS(output_size), THREADS_PER_BLOCK / 2, stream,
                num_pointsets, num_polygons, pointsets.data(),
                polygons.data(), ious.data());
  return diopiSuccess;
}

diopiError_t diopiConvexGiou(diopiContextHandle_t ctx,
                             diopiConstTensorHandle_t pointsets_,
                             diopiConstTensorHandle_t polygons_,
                             diopiTensorHandle_t output_) {
  auto pointsets = impl::cuda::makeTensor(pointsets_);
  auto polygons = impl::cuda::makeTensor(polygons_);
  auto output = impl::cuda::makeTensor(output_);

  int output_size = output.numel();
  int num_pointsets = pointsets.size(0);
  int num_polygons = polygons.size(0);

  // at::cuda::CUDAGuard device_guard(pointsets.device());
  auto stream = impl::cuda::getStream(ctx);
  dispatch_float_types_and_half(
                convex_giou_cuda_kernel,
                pointsets.scalar_type(),
                GET_BLOCKS(output_size), THREADS_PER_BLOCK / 2, stream,
                num_pointsets, num_polygons, pointsets.data(),
                polygons.data(), output.data());
  return diopiSuccess;
}

template <typename T>
__global__ void deform_roi_pool_forward_cuda_kernel(
    const int nthreads, const void* input_, const void* rois_, const void* offset_,
    void* output_, const int pooled_height, const int pooled_width,
    const T spatial_scale, const int sampling_ratio, const T gamma,
    const int channels, const int height, const int width) {
  const T* input = static_cast<const T*>(input_);
  const T* rois = static_cast<const T*>(rois_);
  const T* offset = static_cast<const T*>(offset_);
  T* output = static_cast<T*>(output_);
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_rois[1] * spatial_scale - 0.5;
    T roi_start_h = offset_rois[2] * spatial_scale - 0.5;
    T roi_end_w = offset_rois[3] * spatial_scale - 0.5;
    T roi_end_h = offset_rois[4] * spatial_scale - 0.5;

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_input =
        input + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h =
        (sampling_ratio > 0)
            ? sampling_ratio
            : static_cast<int>(ceilf(roi_height / pooled_height));
    int roi_bin_grid_w =
        (sampling_ratio > 0)
            ? sampling_ratio
            : static_cast<int>(ceilf(roi_width / pooled_width));

    // Compute roi offset
    if (offset != NULL) {
      const T* offset_cur_w = offset + n * pooled_width * pooled_height * 2 +
                              ph * pooled_width + pw;
      T offset_roi_w = gamma * roi_width * offset_cur_w[0];
      T offset_roi_h =
          gamma * roi_height * offset_cur_w[pooled_width * pooled_height];
      roi_start_w += offset_roi_w;
      roi_start_h += offset_roi_h;
    }

    // We do average pooling inside a bin
    const T count = max(roi_bin_grid_h * roi_bin_grid_w, 1);
    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
      const T y = roi_start_h + ph * bin_size_h +
                  static_cast<T>(iy + .5f) * bin_size_h /
                      static_cast<T>(roi_bin_grid_h);
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);
        T val = bilinear_interpolate(offset_input, height, width, y, x, index);
        output_val += val;
      }
    }
    output[index] = output_val / count;
  }
}

template <typename T>
__global__ void deform_roi_pool_backward_cuda_kernel(
    const int nthreads, const void* grad_output_, const void* input_, const void* rois_,
    const void* offset_, void* grad_input_, void* grad_offset_, const int pooled_height,
    const int pooled_width, const T spatial_scale, const int sampling_ratio,
    const T gamma, const int channels, const int height, const int width) {
  const T* grad_output = static_cast<const T*>(grad_output_);
  const T* input = static_cast<const T*>(input_);
  const T* rois = static_cast<const T*>(rois_);
  const T* offset = static_cast<const T*>(offset_);
  T* grad_input = static_cast<T*>(grad_input_);
  T* grad_offset = static_cast<T*>(grad_offset_);
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    const T* offset_input =
        input + ((roi_batch_ind * channels + c) * height * width);
    T* offset_grad_input =
        grad_input + ((roi_batch_ind * channels + c) * height * width);

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_rois[1] * spatial_scale - 0.5;
    T roi_start_h = offset_rois[2] * spatial_scale - 0.5;
    T roi_end_w = offset_rois[3] * spatial_scale - 0.5;
    T roi_end_h = offset_rois[4] * spatial_scale - 0.5;

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h =
        (sampling_ratio > 0)
            ? sampling_ratio
            : static_cast<int>(ceilf(roi_height / pooled_height));
    int roi_bin_grid_w =
        (sampling_ratio > 0)
            ? sampling_ratio
            : static_cast<int>(ceilf(roi_width / pooled_width));

    // Compute roi offset
    if (offset != NULL) {
      const T* offset_cur_w = offset + n * pooled_width * pooled_height * 2 +
                              ph * pooled_width + pw;
      T offset_roi_w = gamma * roi_width * offset_cur_w[0];
      T offset_roi_h =
          gamma * roi_height * offset_cur_w[pooled_width * pooled_height];
      roi_start_w += offset_roi_w;
      roi_start_h += offset_roi_h;
    }

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4
    const T grad_output_this_bin = grad_output[index] / count;

    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
      const T y = roi_start_h + ph * bin_size_h +
                  static_cast<T>(iy + .5f) * bin_size_h /
                      static_cast<T>(roi_bin_grid_h);
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;
        bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4,
                                      x_low, x_high, y_low, y_high, index);

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(offset_grad_input + y_low * width + x_low,
                    grad_output_this_bin * w1);
          atomicAdd(offset_grad_input + y_low * width + x_high,
                    grad_output_this_bin * w2);
          atomicAdd(offset_grad_input + y_high * width + x_low,
                    grad_output_this_bin * w3);
          atomicAdd(offset_grad_input + y_high * width + x_high,
                    grad_output_this_bin * w4);
          if (offset != NULL) {
            T input_00 = offset_input[y_low * width + x_low];
            T input_10 = offset_input[y_low * width + x_high];
            T input_01 = offset_input[y_high * width + x_low];
            T input_11 = offset_input[y_high * width + x_high];
            T ogx = gamma * roi_width * grad_output_this_bin *
                    (input_11 * (y - y_low) + input_10 * (y_high - y) +
                     input_01 * (y_low - y) + input_00 * (y - y_high));
            T ogy = gamma * roi_height * grad_output_this_bin *
                    (input_11 * (x - x_low) + input_01 * (x_high - x) +
                     input_10 * (x_low - x) + input_00 * (x - x_high));
            atomicAdd(grad_offset + n * pooled_width * pooled_height * 2 +
                          ph * pooled_width + pw,
                      ogx);
            atomicAdd(grad_offset + n * pooled_width * pooled_height * 2 +
                          pooled_width * pooled_height + ph * pooled_width + pw,
                      ogy);
          }
        }
      }
    }
  }
}

diopiError_t diopiDeformRoiPool(diopiContextHandle_t ctx, diopiTensorHandle_t input_,
                   diopiTensorHandle_t rois_, diopiTensorHandle_t offset_,
                   diopiTensorHandle_t output_, int64_t pooled_height,
                   int64_t pooled_width, float spatial_scale,
                   int64_t sampling_ratio, float gamma) {
  auto input = impl::cuda::makeTensor(input_);
  auto rois = impl::cuda::makeTensor(rois_);
  auto offset = impl::cuda::makeTensor(offset_);
  auto output = impl::cuda::makeTensor(output_);
  int output_size = output.numel();
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  // at::cuda::CUDAGuard device_guard(input.device());
  auto stream = impl::cuda::getStream(ctx);
  // dispatch_float_types_and_half(
  //               deform_roi_pool_forward_cuda_kernel,
  //               input.scalar_type(),
  //               GET_BLOCKS(output_size), THREADS_PER_BLOCK, stream,
  //               output_size, input.data(),
  //               rois.data(), offset.data(),
  //               output.data(), pooled_height, pooled_width,
  //               static_cast<scalar_t>(spatial_scale), sampling_ratio,
  //               static_cast<scalar_t>(gamma), channels, height, width);

  // 这里在double情形下，不知道有没有问题？是否会自动提升
  dispatch_float_types_and_half(
                deform_roi_pool_forward_cuda_kernel,
                input.scalar_type(),
                GET_BLOCKS(output_size), THREADS_PER_BLOCK, stream,
                output_size, input.data(),
                rois.data(), offset.data(),
                output.data(), pooled_height, pooled_width,
                spatial_scale, sampling_ratio,
                gamma, channels, height, width);

  return diopiSuccess;
}

diopiError_t diopiDeformRoiPoolBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output_,
    diopiTensorHandle_t input_, diopiTensorHandle_t rois_,
    diopiTensorHandle_t offset_, diopiTensorHandle_t grad_input_,
    diopiTensorHandle_t grad_offset_, int64_t pooled_height,
    int64_t pooled_width, float spatial_scale, int64_t sampling_ratio,
    float gamma) {
  auto grad_output = impl::cuda::makeTensor(grad_output_);
  auto input = impl::cuda::makeTensor(input_);
  auto rois = impl::cuda::makeTensor(rois_);
  auto offset = impl::cuda::makeTensor(offset_);
  auto grad_input = impl::cuda::makeTensor(grad_input_);
  auto grad_offset = impl::cuda::makeTensor(grad_offset_);

  int output_size = grad_output.numel();
  int channels = grad_input.size(1);
  int height = grad_input.size(2);
  int width = grad_input.size(3);

  // at::cuda::CUDAGuard device_guard(grad_output.device());
  auto stream = impl::cuda::getStream(ctx);
  // dispatch_float_types_and_half(
  //               deform_roi_pool_backward_cuda_kernel,
  //               grad_output.scalar_type(),
  //               GET_BLOCKS(output_size), THREADS_PER_BLOCK, stream,
  //               output_size, grad_output.data(),
  //               input.data(), rois.data(),
  //               offset.data(), grad_input.data(),
  //               grad_offset.data(), pooled_height, pooled_width,
  //               static_cast<scalar_t>(spatial_scale), sampling_ratio,
  //               static_cast<scalar_t>(gamma), channels, height, width);
  dispatch_float_types_and_half(
                deform_roi_pool_backward_cuda_kernel,
                grad_output.scalar_type(),
                GET_BLOCKS(output_size), THREADS_PER_BLOCK, stream,
                output_size, grad_output.data(),
                input.data(), rois.data(),
                offset.data(), grad_input.data(),
                grad_offset.data(), pooled_height, pooled_width,
                spatial_scale, sampling_ratio,
                gamma, channels, height, width);
  return diopiSuccess;
}

inline __device__ void swap_float(float *x, float *y) {
  float tmp = *x;
  *x = *y;
  *y = tmp;
}

inline __device__ void swap_int(int *x, int *y) {
  int tmp = *x;
  *x = *y;
  *y = tmp;
}

__device__ void reheap(float *dist, int *idx, int k) {
  int root = 0;
  int child = root * 2 + 1;
  while (child < k) {
    if (child + 1 < k && dist[child + 1] > dist[child]) child++;
    if (dist[root] > dist[child]) return;
    swap_float(&dist[root], &dist[child]);
    swap_int(&idx[root], &idx[child]);
    root = child;
    child = root * 2 + 1;
  }
}

__device__ void heap_sort(float *dist, int *idx, int k) {
  int i;
  for (i = k - 1; i > 0; i--) {
    swap_float(&dist[0], &dist[i]);
    swap_int(&idx[0], &idx[i]);
    reheap(dist, idx, i);
  }
}

// input: xyz (b, n, 3) new_xyz (b, m, 3)
// output: idx (b, m, nsample) dist2 (b, m, nsample)
template <typename T>
__global__ void knn_forward_cuda_kernel(int b, int n, int m, int nsample,
                                        const void *xyz_, const void *new_xyz_,
                                        int *__restrict__ idx, void *dist2_) {
  int bs_idx = blockIdx.y;
  const T* xyz = static_cast<const T*>(xyz_);
  const T* new_xyz = static_cast<const T*>(new_xyz_);
  T* dist2 = static_cast<T*>(dist2_);
  CUDA_1D_KERNEL_LOOP(pt_idx, m) {
    if (bs_idx >= b) return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    idx += bs_idx * m * nsample + pt_idx * nsample;
    dist2 += bs_idx * m * nsample + pt_idx * nsample;

    T new_x = new_xyz[0];
    T new_y = new_xyz[1];
    T new_z = new_xyz[2];

    float best_dist[100];
    int best_idx[100];
    for (int i = 0; i < nsample; i++) {
      best_dist[i] = 1e10;
      best_idx[i] = 0;
    }
    for (int i = 0; i < n; i++) {
      T x = xyz[i * 3 + 0];
      T y = xyz[i * 3 + 1];
      T z = xyz[i * 3 + 2];
      T d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
             (new_z - z) * (new_z - z);
      if (d2 < best_dist[0]) {
        best_dist[0] = d2;
        best_idx[0] = i;
        reheap(best_dist, best_idx, nsample);
      }
    }
    heap_sort(best_dist, best_idx, nsample);
    for (int i = 0; i < nsample; i++) {
      idx[i] = best_idx[i];
      dist2[i] = best_dist[i];
    }
  }
}

diopiError_t diopiKnn(diopiContextHandle_t ctx, diopiTensorHandle_t xyz_,
                      diopiTensorHandle_t new_xyz_,
                      diopiTensorHandle_t idx_,
                      diopiTensorHandle_t dist2_, int64_t b, int64_t n,
                      int64_t m, int64_t nsample) {
  // param new_xyz: (B, m, 3)
  // param xyz: (B, n, 3)
  // param idx: (B, m, nsample)

  auto xyz = impl::cuda::makeTensor(xyz_);
  auto new_xyz = impl::cuda::makeTensor(new_xyz_);
  auto idx = impl::cuda::makeTensor(idx_);
  auto dist2 = impl::cuda::makeTensor(dist2_);

  // at::cuda::CUDAGuard device_guard(new_xyz.device());
  auto stream = impl::cuda::getStream(ctx);

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(GET_BLOCKS(m, THREADS_PER_BLOCK), b);
  dim3 threads(THREADS_PER_BLOCK);

  dispatch_float_types_and_half(
            knn_forward_cuda_kernel,
            new_xyz.scalar_type(),
            blocks, threads, stream,
            b, n, m, nsample, xyz.data(),
            new_xyz.data(), static_cast<int*>(idx.data()),
            dist2.data());

  return diopiSuccess;
}
