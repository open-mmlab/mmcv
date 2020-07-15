#ifndef NMS_CUDA_KERNEL_CUH
#define NMS_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long int) * 8;

__device__ inline bool devIoU(float const *const a, float const *const b,
                              const int offset, const float threshold) {
  float left = fmaxf(a[0], b[0]), right = fminf(a[2], b[2]);
  float top = fmaxf(a[1], b[1]), bottom = fminf(a[3], b[3]);
  float width = fmaxf(right - left + offset, 0.f),
        height = fmaxf(bottom - top + offset, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + offset) * (a[3] - a[1] + offset);
  float Sb = (b[2] - b[0] + offset) * (b[3] - b[1] + offset);
  return interS > threshold * (Sa + Sb - interS);
}

__global__ void nms_cuda(const int n_boxes, const float iou_threshold,
                         const int offset, const float *dev_boxes,
                         unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;
  const int tid = threadIdx.x;

  if (row_start > col_start) return;

  const int row_size =
      fminf(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
      fminf(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 4];
  if (tid < col_size) {
    block_boxes[tid * 4 + 0] =
        dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 0];
    block_boxes[tid * 4 + 1] =
        dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 1];
    block_boxes[tid * 4 + 2] =
        dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 2];
    block_boxes[tid * 4 + 3] =
        dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 3];
  }
  __syncthreads();

  if (tid < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + tid;
    const float *cur_box = dev_boxes + cur_box_idx * 4;
    int i = 0;
    unsigned long long int t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = tid + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 4, offset, iou_threshold)) {
        t |= 1ULL << i;
      }
    }
    dev_mask[cur_box_idx * gridDim.y + col_start] = t;
  }
}
#endif  // NMS_CUDA_KERNEL_CUH
