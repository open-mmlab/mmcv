#ifndef BBOX_OVERLAPS_CUDA_KERNEL_CUH
#define BBOX_OVERLAPS_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename T>
__global__ void bbox_overlaps_cuda_kernel(const T* bbox1, const T* bbox2,
                                          T* ious, const int num_bbox1,
                                          const int num_bbox2, const int mode,
                                          const bool aligned,
                                          const int offset) {
  if (aligned) {
    CUDA_1D_KERNEL_LOOP(index, num_bbox1) {
      int b1 = index;
      int b2 = index;

      int base1 = b1 * 4;
      T b1_x1 = bbox1[base1];
      T b1_y1 = bbox1[base1 + 1];
      T b1_x2 = bbox1[base1 + 2];
      T b1_y2 = bbox1[base1 + 3];
      T b1_area = (b1_x2 - b1_x1 + offset) * (b1_y2 - b1_y1 + offset);

      int base2 = b2 * 4;
      T b2_x1 = bbox2[base2];
      T b2_y1 = bbox2[base2 + 1];
      T b2_x2 = bbox2[base2 + 2];
      T b2_y2 = bbox2[base2 + 3];
      T b2_area = (b2_x2 - b2_x1 + offset) * (b2_y2 - b2_y1 + offset);

      T left = fmaxf(b1_x1, b2_x1), right = fminf(b1_x2, b2_x2);
      T top = fmaxf(b1_y1, b2_y1), bottom = fminf(b1_y2, b2_y2);
      T width = fmaxf(right - left + offset, 0.f);
      T height = fmaxf(bottom - top + offset, 0.f);
      T interS = width * height;
      T baseS = 1.0;
      if (mode == 0) {
        baseS = fmaxf(b1_area + b2_area - interS, T(offset));
      } else if (mode == 1) {
        baseS = fmaxf(b1_area, T(offset));
      }
      ious[index] = interS / baseS;
    }
  } else {
    CUDA_1D_KERNEL_LOOP(index, num_bbox1 * num_bbox2) {
      int b1 = index / num_bbox2;
      int b2 = index % num_bbox2;

      int base1 = b1 * 4;
      T b1_x1 = bbox1[base1];
      T b1_y1 = bbox1[base1 + 1];
      T b1_x2 = bbox1[base1 + 2];
      T b1_y2 = bbox1[base1 + 3];
      T b1_area = (b1_x2 - b1_x1 + offset) * (b1_y2 - b1_y1 + offset);

      int base2 = b2 * 4;
      T b2_x1 = bbox2[base2];
      T b2_y1 = bbox2[base2 + 1];
      T b2_x2 = bbox2[base2 + 2];
      T b2_y2 = bbox2[base2 + 3];
      T b2_area = (b2_x2 - b2_x1 + offset) * (b2_y2 - b2_y1 + offset);

      T left = fmaxf(b1_x1, b2_x1), right = fminf(b1_x2, b2_x2);
      T top = fmaxf(b1_y1, b2_y1), bottom = fminf(b1_y2, b2_y2);
      T width = fmaxf(right - left + offset, 0.f);
      T height = fmaxf(bottom - top + offset, 0.f);
      T interS = width * height;
      T baseS = 1.0;
      if (mode == 0) {
        baseS = fmaxf(b1_area + b2_area - interS, T(offset));
      } else if (mode == 1) {
        baseS = fmaxf(b1_area, T(offset));
      }
      ious[index] = interS / baseS;
    }
  }
}

#endif  // BBOX_OVERLAPS_CUDA_KERNEL_CUH
