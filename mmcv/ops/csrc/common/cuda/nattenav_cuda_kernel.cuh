/*!
**************************************************************************************************
* NATTEN-COMMON FUNCTIONS (CUDA)
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from
*https://github.com/SHI-Labs/Neighborhood-Attention-Transformer
**************************************************************************************************
*/
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define AT_DISPATCH_HALF_TYPES(SCALARTYPE1, TYPE, NAME, ...)                \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                \
    switch (_st) {                                                          \
      AT_PRIVATE_CASE_TYPE(                                                 \
          NAME, SCALARTYPE1,                                                \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),         \
          __VA_ARGS__)                                                      \
      default:                                                              \
        break;                                                              \
    }                                                                       \
  }()

#define CUDA_NUM_THREADS 1024

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int64_t N,
                      const int64_t max_threads_per_block = CUDA_NUM_THREADS) {
  auto block_num = (N - 1) / max_threads_per_block + 1;
  return static_cast<int>(block_num);
}

inline __host__ __device__ int get_backward_window_start(
    const int index, const int KERNEL_SIZE, const int NEIGHBORHOOD_SIZE) {
  return (index < KERNEL_SIZE) ? (0) : index - NEIGHBORHOOD_SIZE;
}

inline __host__ __device__ int get_backward_window_end(
    const int index, const int length, const int KERNEL_SIZE,
    const int NEIGHBORHOOD_SIZE) {
  return (index >= length - KERNEL_SIZE) ? (length)
                                         : (index + (NEIGHBORHOOD_SIZE + 1));
}

inline __host__ __device__ int get_window_start(const int index,
                                                const int length,
                                                const int KERNEL_SIZE,
                                                const int NEIGHBORHOOD_SIZE) {
  return max(index - NEIGHBORHOOD_SIZE, 0) +
         (index + NEIGHBORHOOD_SIZE >= length) *
             (length - index - NEIGHBORHOOD_SIZE - 1);
}

inline __host__ __device__ int get_pb_start(const int index, const int length,
                                            const int KERNEL_SIZE,
                                            const int NEIGHBORHOOD_SIZE) {
  return NEIGHBORHOOD_SIZE +
         (index < NEIGHBORHOOD_SIZE) * (NEIGHBORHOOD_SIZE - index) +
         (index + NEIGHBORHOOD_SIZE >= length) *
             (length - index - 1 - NEIGHBORHOOD_SIZE);
}

#define CHECK_SEQUENCE(length, kernel_size) \
  TORCH_CHECK(                              \
      length >= kernel_size,                \
      "Input sequence length must be greater than or equal to kernel size.")
#define CHECK_FEATMAP(height, width, kernel_size)    \
  TORCH_CHECK(                                       \
      height >= kernel_size && width >= kernel_size, \
      "Input resolution must be greater than or equal to kernel size.")

// 2D Neighborhood Attention

// THE FOLLOWING CAN BE MODIFIED TO SUPPORT ADDITIONAL KERNEL SIZES
// MAKE SURE TO EDIT BOTH CHECK_KERNELSIZE AND LAUNCH_DNA_KNS

#define CHECK_KERNELSIZE(NAME, kernel_size)                                   \
  TORCH_CHECK(kernel_size == 3 || kernel_size == 5 || kernel_size == 7 ||     \
                  kernel_size == 9 || kernel_size == 11 || kernel_size == 13, \
              NAME, " does not support kernel size ", kernel_size)

// 2D KERNEL LAUNCHER
// First number is the kernel size itself, second is floor(kernel_size / 2) aka
// neighborhood radius.
#define LAUNCH_DNA_KNS(kernel_size, NAME, BLK, TPB, SMEM, CSTREAM, ...)  \
  ({                                                                     \
    switch (kernel_size) {                                               \
      case 3:                                                            \
        NAME<3, 1, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);  \
        break;                                                           \
      case 5:                                                            \
        NAME<5, 2, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);  \
        break;                                                           \
      case 7:                                                            \
        NAME<7, 3, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);  \
        break;                                                           \
      case 9:                                                            \
        NAME<9, 4, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);  \
        break;                                                           \
      case 11:                                                           \
        NAME<11, 5, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__); \
        break;                                                           \
      case 13:                                                           \
        NAME<13, 6, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__); \
        break;                                                           \
      default:                                                           \
        TORCH_INTERNAL_ASSERT(false);                                    \
        break;                                                           \
    }                                                                    \
  })

// 1D KERNEL LAUNCHER
// First number is the kernel size itself, second is floor(kernel_size / 2) aka
// neighborhood radius.
#define LAUNCH_DNA_KNS_1D(kernel_size, NAME, BLK, TPB, SMEM, CSTREAM, ...) \
  ({                                                                       \
    switch (kernel_size) {                                                 \
      case 3:                                                              \
        NAME<3, 1, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);    \
        break;                                                             \
      case 5:                                                              \
        NAME<5, 2, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);    \
        break;                                                             \
      case 7:                                                              \
        NAME<7, 3, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);    \
        break;                                                             \
      case 9:                                                              \
        NAME<9, 4, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);    \
        break;                                                             \
      case 11:                                                             \
        NAME<11, 5, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);   \
        break;                                                             \
      case 13:                                                             \
        NAME<13, 6, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);   \
        break;                                                             \
      default:                                                             \
        NAME<-1, -1, scalar_t><<<BLK, TPB, SMEM, CSTREAM>>>(__VA_ARGS__);  \
        break;                                                             \
    }                                                                      \
  })
