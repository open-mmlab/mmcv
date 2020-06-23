#ifndef SOFTNMS_KERNEL_CUH
#define SOFTNMS_KERNEL_CUH

#include <cuda.h>

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long int) * 8;

template <typename scalar_t>
__device__ inline scalar_t devIoU(scalar_t const *const a,
                                  scalar_t const *const b) {
  scalar_t left = fmaxf(a[0], b[0]), right = fminf(a[2], b[2]);
  scalar_t top = fmaxf(a[1], b[1]), bottom = fminf(a[3], b[3]);
  scalar_t width = fmaxf(right - left + 1, 0.f),
           height = fmaxf(bottom - top + 1, 0.f);
  scalar_t interS = width * height;
  scalar_t Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  scalar_t Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

template <typename scalar_t>
__global__ void softnms_max_kernel(const int n_boxes,
                                   const scalar_t overlap_thresh,
                                   const scalar_t *dev_boxes, int *order,
                                   float *max_value, int *max_index) {
  __shared__ float maximum[threadsPerBlock];
  __shared__ int max_id[threadsPerBlock];

  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * threadsPerBlock + threadIdx.x;

  if (idx >= n_boxes) {
    return;
  }

  const int block_size = fminf(n_boxes + tid - idx, threadsPerBlock);
  int *l_order = order + (idx - tid);
  if (l_order[tid] == 0 && dev_boxes[idx * 5 + 4] >= overlap_thresh) {
    maximum[tid] = dev_boxes[idx * 5 + 4];
  } else {
    maximum[tid] = -1.0;
  }
  max_id[tid] = tid;
  __syncthreads();

  if (block_size >= 1024 && tid < 512) {
    if (maximum[tid] < maximum[tid + 512]) {
      maximum[tid] = maximum[tid + 512];
      max_id[tid] = max_id[tid + 512];
    }
  }
  if (block_size >= 512 && tid < 256) {
    if (maximum[tid] < maximum[tid + 256]) {
      maximum[tid] = maximum[tid + 256];
      max_id[tid] = max_id[tid + 256];
    }
  }
  if (block_size >= 256 && tid < 128) {
    if (maximum[tid] < maximum[tid + 128]) {
      maximum[tid] = maximum[tid + 128];
      max_id[tid] = max_id[tid + 128];
    }
  }
  if (block_size >= 128 && tid < 64) {
    if (maximum[tid] < maximum[tid + 64]) {
      maximum[tid] = maximum[tid + 64];
      max_id[tid] = max_id[tid + 64];
    }
  }
  if (tid < 32) {
    volatile float *vmaximum = maximum;
    volatile int *vmax_id = max_id;
    if (block_size >= 64 && vmaximum[tid] < vmaximum[tid + 32]) {
      vmaximum[tid] = vmaximum[tid + 32];
      vmax_id[tid] = vmax_id[tid + 32];
    }
    if (block_size >= 32 && tid < 16 && vmaximum[tid] < vmaximum[tid + 16]) {
      vmaximum[tid] = vmaximum[tid + 16];
      vmax_id[tid] = vmax_id[tid + 16];
    }
    if (block_size >= 16 && tid < 8 && vmaximum[tid] < vmaximum[tid + 8]) {
      vmaximum[tid] = vmaximum[tid + 8];
      vmax_id[tid] = vmax_id[tid + 8];
    }
    if (block_size >= 8 && tid < 4 && vmaximum[tid] < vmaximum[tid + 4]) {
      vmaximum[tid] = vmaximum[tid + 4];
      vmax_id[tid] = vmax_id[tid + 4];
    }
    if (block_size >= 4 && tid < 2 && vmaximum[tid] < vmaximum[tid + 2]) {
      vmaximum[tid] = vmaximum[tid + 2];
      vmax_id[tid] = vmax_id[tid + 2];
    }
    if (block_size >= 2 && tid < 1 && vmaximum[tid] < vmaximum[tid + 1]) {
      vmaximum[tid] = vmaximum[tid + 1];
      vmax_id[tid] = vmax_id[tid + 1];
    }
  }
  if (tid == 0) {
    max_value[blockIdx.x] = maximum[0];
    max_index[blockIdx.x] = max_id[0];
  }
}

template <typename scalar_t>
__global__ void softnms_update_kernel(const int n_boxes, const scalar_t sigma,
                                      const scalar_t n_thresh,
                                      const unsigned int method,
                                      const scalar_t overlap_thresh,
                                      scalar_t *dev_boxes, int *order,
                                      unsigned long long *keep, int max_id) {
  const int col_start = blockIdx.x;

  const int col_size =
      fminf(n_boxes - col_start * threadsPerBlock, threadsPerBlock);
  const int cur_idx = threadsPerBlock * col_start + threadIdx.x;
  const int tid = threadIdx.x;

  if (cur_idx >= n_boxes) {
    return;
  }
  __shared__ scalar_t cur_max_boxes[5];

  cur_max_boxes[0] = dev_boxes[max_id * 5 + 0];
  cur_max_boxes[1] = dev_boxes[max_id * 5 + 1];
  cur_max_boxes[2] = dev_boxes[max_id * 5 + 2];
  cur_max_boxes[3] = dev_boxes[max_id * 5 + 3];
  cur_max_boxes[4] = dev_boxes[max_id * 5 + 4];

  __syncthreads();

  if (cur_idx != max_id && tid < col_size && order[cur_idx] == 0 &&
      (!(keep[col_start] & (1ULL << tid)))) {
    scalar_t block_boxes[5];
    block_boxes[0] = dev_boxes[cur_idx * 5 + 0];
    block_boxes[1] = dev_boxes[cur_idx * 5 + 1];
    block_boxes[2] = dev_boxes[cur_idx * 5 + 2];
    block_boxes[3] = dev_boxes[cur_idx * 5 + 3];
    block_boxes[4] = dev_boxes[cur_idx * 5 + 4];

    scalar_t ovr = devIoU(cur_max_boxes, block_boxes);
    scalar_t weight = 1.0;
    if (method == 1) {
      if (ovr > n_thresh) {
        weight = 1.0 - ovr;
      }
    } else if (method == 2) {
      weight = exp(-(ovr * ovr) / sigma);
    } else if (ovr >= n_thresh) {
      weight = 0.0;
    }
    block_boxes[4] *= weight;
    dev_boxes[cur_idx * 5 + 4] = block_boxes[4];
    if (block_boxes[4] < overlap_thresh) {
      keep[col_start] |= 1ULL << tid;
    }
  }
}

#endif
