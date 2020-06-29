#ifndef CC_ATTENTION_CUDA_KERNEL_CUH
#define CC_ATTENTION_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename T>
__global__ void ca_forward_kernel(const T *t, const T *f, T *weight, int num,
                                  int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  int len = height + width - 1;
  int z = blockIdx.z;

  if (x < width && y < height && z < height + width - 1) {
    for (int batch = 0; batch < num; ++batch) {
      for (int plane = 0; plane < chn; ++plane) {
        T _t = t[(batch * chn + plane) * sp + y * width + x];

        if (z < width) {
          int i = z;
          T _f = f[(batch * chn + plane) * sp + y * width + i];
          weight[(batch * len + i) * sp + y * width + x] += _t * _f;
        } else {
          int i = z - width;
          int j = i < y ? i : i + 1;

          T _f = f[(batch * chn + plane) * sp + j * width + x];
          weight[(batch * len + width + i) * sp + y * width + x] += _t * _f;
        }
      }
    }
  }
}

template <typename T>
__global__ void ca_backward_kernel_t(const T *dw, const T *t, const T *f, T *dt,
                                     int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  int len = height + width - 1;
  int plane = blockIdx.z;

  if (x < width && y < height && plane < chn) {
    for (int batch = 0; batch < num; ++batch) {
      for (int i = 0; i < width; ++i) {
        T _dw = dw[(batch * len + i) * sp + y * width + x];
        T _f = f[(batch * chn + plane) * sp + y * width + i];
        dt[(batch * chn + plane) * sp + y * width + x] += _dw * _f;
      }
      for (int i = 0; i < height; ++i) {
        if (i == y) continue;
        int j = i < y ? i : i - 1;

        T _dw = dw[(batch * len + width + j) * sp + y * width + x];
        T _f = f[(batch * chn + plane) * sp + i * width + x];
        dt[(batch * chn + plane) * sp + y * width + x] += _dw * _f;
      }
    }
  }
}

template <typename T>
__global__ void ca_backward_kernel_f(const T *dw, const T *t, const T *f, T *df,
                                     int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  int len = height + width - 1;
  int plane = blockIdx.z;

  if (x < width && y < height && plane < chn) {
    for (int batch = 0; batch < num; ++batch) {
      for (int i = 0; i < width; ++i) {
        T _dw = dw[(batch * len + x) * sp + y * width + i];
        T _t = t[(batch * chn + plane) * sp + y * width + i];
        df[(batch * chn + plane) * sp + y * width + x] += _dw * _t;
      }
      for (int i = 0; i < height; ++i) {
        if (i == y) continue;
        int j = i > y ? y : y - 1;

        T _dw = dw[(batch * len + width + j) * sp + i * width + x];
        T _t = t[(batch * chn + plane) * sp + i * width + x];
        df[(batch * chn + plane) * sp + y * width + x] += _dw * _t;
      }
    }
  }
}

template <typename T>
__global__ void ca_map_forward_kernel(const T *weight, const T *g, T *out,
                                      int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  int len = height + width - 1;
  int plane = blockIdx.z;

  if (x < width && y < height && plane < chn) {
    for (int batch = 0; batch < num; ++batch) {
      for (int i = 0; i < width; ++i) {
        T _g = g[(batch * chn + plane) * sp + y * width + i];
        T _w = weight[(batch * len + i) * sp + y * width + x];
        out[(batch * chn + plane) * sp + y * width + x] += _g * _w;
      }
      for (int i = 0; i < height; ++i) {
        if (i == y) continue;

        int j = i < y ? i : i - 1;

        T _g = g[(batch * chn + plane) * sp + i * width + x];
        T _w = weight[(batch * len + width + j) * sp + y * width + x];
        out[(batch * chn + plane) * sp + y * width + x] += _g * _w;
      }
    }
  }
}

template <typename T>
__global__ void ca_map_backward_kernel_w(const T *dout, const T *weight,
                                         const T *g, T *dw, int num, int chn,
                                         int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  int len = height + width - 1;
  int z = blockIdx.z;

  if (x < width && y < height && z < height + width - 1) {
    for (int batch = 0; batch < num; ++batch) {
      for (int plane = 0; plane < chn; ++plane) {
        T _dout = dout[(batch * chn + plane) * sp + y * width + x];

        if (z < width) {
          int i = z;
          T _g = g[(batch * chn + plane) * sp + y * width + i];
          dw[(batch * len + i) * sp + y * width + x] += _dout * _g;
        } else {
          int i = z - width;
          int j = i < y ? i : i + 1;

          T _g = g[(batch * chn + plane) * sp + j * width + x];
          dw[(batch * len + width + i) * sp + y * width + x] += _dout * _g;
        }
      }
    }
  }
}

template <typename T>
__global__ void ca_map_backward_kernel_g(const T *dout, const T *weight,
                                         const T *g, T *dg, int num, int chn,
                                         int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  int len = height + width - 1;
  int plane = blockIdx.z;

  if (x < width && y < height && plane < chn) {
    for (int batch = 0; batch < num; ++batch) {
      for (int i = 0; i < width; ++i) {
        T _dout = dout[(batch * chn + plane) * sp + y * width + i];
        T _w = weight[(batch * len + x) * sp + y * width + i];
        dg[(batch * chn + plane) * sp + y * width + x] += _dout * _w;
      }
      for (int i = 0; i < height; ++i) {
        if (i == y) continue;
        int j = i > y ? y : y - 1;

        T _dout = dout[(batch * chn + plane) * sp + i * width + x];
        T _w = weight[(batch * len + width + j) * sp + i * width + x];
        dg[(batch * chn + plane) * sp + y * width + x] += _dout * _w;
      }
    }
  }
}

#endif  // CC_ATTENTION_CUDA_KERNEL_CUH
