// Copyright (c) OpenMMLab. All rights reserved.
#ifndef RANS_CUDA_KERNEL_CUH
#define RANS_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

#include "utils/rans/rans.hpp"

#define BYTES_OF(x) sizeof(decltype(x))
#define CUDA_MALLOC(x, num) cudaMalloc((void**)&x, num* BYTES_OF(*x))
#define CUDA_MEMSET(x, num, value) cudaMemset(x, value, num* BYTES_OF(*x))
#define CPU_TO_CUDA(x_cpu, x_cuda, num) \
  cudaMemcpy(x_cuda, x_cpu, num* BYTES_OF(*x_cuda), cudaMemcpyHostToDevice)
#define CREATE_TENSOR(shape, type, target_device) \
  zeros(shape, TensorOptions().dtype(type).device(target_device));
#define CUDA_FREE(x) cudaFree(x)

struct RansCUDAKernelResult {
  uint32_t raw_value;
  bool bypass_mode;
};

// ---------------------------- Encoding -------------------------------------
__global__ void rans_encode_with_indexes_cuda_kernel(
    const int* symbols, const int* indexes, const int* cdfs,
    const int* cdfs_sizes, const int* offsets, const int num_symbols,
    const int num_cdfs, const int max_cdf_bin_size, RansSymbol* rans_symbols,
    RansCUDAKernelResult* results) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < num_symbols) {
    const size_t cdf_idx = indexes[i];
    assert(cdf_idx < num_cdfs);

    const int* cdf = cdfs + cdf_idx * max_cdf_bin_size;

    const int32_t max_value = cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0 && (max_value + 1) < max_cdf_bin_size);

    int32_t value = symbols[i] - offsets[cdf_idx];

    uint32_t raw_val = 0;
    if (value < 0) {
      raw_val = -2 * value - 1;
      value = max_value;
    } else if (value >= max_value) {
      raw_val = 2 * (value - max_value);
      value = max_value;
    }

    assert(value >= 0 && value < max_value + 1);

    rans_symbols[i].start = cdf[value];
    rans_symbols[i].range = cdf[value + 1] - cdf[value];
    assert(rans_symbols[i].range != 0);
    results[i].raw_value = raw_val;
    results[i].bypass_mode = (value == max_value);
  }
}
// ------------------- PMF TO QUANTIZED CDF -------------------

__global__ void pmf_to_quantized_cdf_cuda_kernel(
    const float* pmfs, const int* pmf_lengths, const float* tail_masses,
    int* quantized_cdfs, const int pmfs_size0, const int pmfs_size1,
    const int cdf_max_length) {
  const int i = blockIdx.x;
  const int j = threadIdx.x;
  const int pmf_length = pmf_lengths[i];
  if (i >= pmfs_size0 || j >= pmf_length + 2) return;

  const float* pmf = pmfs + i * pmfs_size1;
  const float tail_mass = tail_masses[i];
  __shared__ int shared_cdf[THREADS_PER_BLOCK];

  if (j == 0) {
    shared_cdf[j] = 0;
  } else if (j == pmf_length + 1) {
    shared_cdf[j] = round(tail_mass * (1 << precision)); /* freq of tail mass */
  } else {
    shared_cdf[j] = round(pmf[j - 1] * (1 << precision));
  }

  __shared__ uint32_t total;
  if (j == 0) {
    total = 0;
  }
  __syncthreads();
  atomicAdd(&total, shared_cdf[j]);
  __syncthreads();
  assert(total != 0);

  shared_cdf[j] =
      (static_cast<uint64_t>(1 << precision) * ((uint32_t)shared_cdf[j])) /
      total;
  __syncthreads();

  // TODO: this could be accelerated by using a block scan
  if (j == 0) {
    for (int k = 1; k < pmf_length + 1; k++) {
      shared_cdf[k] += shared_cdf[k - 1];
    }
    shared_cdf[pmf_length + 1] = 1 << precision;
  }
  __syncthreads();

  if (j == 0) {
    for (int k = 0; k < static_cast<int>(pmf_length + 1); ++k) {
      if (shared_cdf[k] == shared_cdf[k + 1]) {
        /* Try to steal frequency from low-frequency symbols */
        uint32_t best_freq = ~0u;
        int best_steal = -1;
        for (int m = 0; m < static_cast<int>(pmf_length + 2) - 1; ++m) {
          uint32_t freq = shared_cdf[m + 1] - shared_cdf[m];
          if (freq > 1 && freq < best_freq) {
            best_freq = freq;
            best_steal = m;
          }
        }

        assert(best_steal != -1);

        if (best_steal < k) {
          for (int m = best_steal + 1; m <= k; ++m) {
            shared_cdf[m]--;
          }
        } else {
          assert(best_steal > k);
          for (int m = k + 1; m <= best_steal; ++m) {
            shared_cdf[m]++;
          }
        }
      }
    }
  }
  __syncthreads();

  if (j > 0) {
    assert(shared_cdf[j] > shared_cdf[j - 1]);
  }
  if (j == pmf_length + 1) {
    assert(shared_cdf[j] == (1 << precision));
  }

  int* quantized_cdf = quantized_cdfs + i * cdf_max_length;
  quantized_cdf[j] = shared_cdf[j];
}
#endif  // RANS_CUDA_KERNEL_CUH
