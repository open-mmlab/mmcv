// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
#include <c10/util/Half.h>
#include <torch/types.h>

#include "pytorch_musa_helper.hpp"
#if MUSA_ARCH > 21
struct upfirdn2d_kernel_params {
  const void *x;
  const float *f;
  void *y;

  int2 up;
  int2 down;
  int2 pad0;
  int flip;
  float gain;

  int4 inSize;  // [width, height, channel, batch]
  int4 inStride;
  int2 filterSize;  // [width, height]
  int2 filterStride;
  int4 outSize;  // [width, height, channel, batch]
  int4 outStride;
  int sizeMinor;
  int sizeMajor;

  int loopMinor;
  int loopMajor;
  int loopX;
  int launchMinor;
  int launchMajor;
};

//------------------------------------------------------------------------
// MUSA kernel specialization.

struct upfirdn2d_kernel_spec {
  void *kernel;
  int tileOutW;
  int tileOutH;
  int loopMinor;
  int loopX;
};

//------------------------------------------------------------------------
// MUSA kernel selection.

template <class T>
upfirdn2d_kernel_spec choose_upfirdn2d_kernel(const upfirdn2d_kernel_params &p);
//------------------------------------------------------------------------

// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

//------------------------------------------------------------------------
// Helpers.

template <class T>
struct InternalType;
template <>
struct InternalType<double> {
  typedef double scalar_t;
};
template <>
struct InternalType<float> {
  typedef float scalar_t;
};
template <>
struct InternalType<c10::Half> {
  typedef float scalar_t;
};

static __device__ __forceinline__ int floor_div(int a, int b) {
  int t = 1 - a / b;
  return (a + t * b) / b - t;
}

//------------------------------------------------------------------------
// Generic MUSA implementation for large filters.

template <class T>
static __global__ void upfirdn2d_kernel_large(upfirdn2d_kernel_params p) {
  typedef typename InternalType<T>::scalar_t scalar_t;

  // Calculate thread index.
  int minorBase = blockIdx.x * blockDim.x + threadIdx.x;
  int outY = minorBase / p.launchMinor;
  minorBase -= outY * p.launchMinor;
  int outXBase = blockIdx.y * p.loopX * blockDim.y + threadIdx.y;
  int majorBase = blockIdx.z * p.loopMajor;
  if (outXBase >= p.outSize.x | outY >= p.outSize.y | majorBase >= p.sizeMajor)
    return;

  // Setup Y receptive field.
  int midY = outY * p.down.y + p.up.y - 1 - p.pad0.y;
  int inY = min(max(floor_div(midY, p.up.y), 0), p.inSize.y);
  int h =
      min(max(floor_div(midY + p.filterSize.y, p.up.y), 0), p.inSize.y) - inY;
  int filterY = midY + p.filterSize.y - (inY + 1) * p.up.y;
  if (p.flip) filterY = p.filterSize.y - 1 - filterY;

  // Loop over major, minor, and X.
  for (int majorIdx = 0, major = majorBase;
       majorIdx < p.loopMajor & major < p.sizeMajor; majorIdx++, major++)
    for (int minorIdx = 0, minor = minorBase;
         minorIdx < p.loopMinor & minor < p.sizeMinor;
         minorIdx++, minor += p.launchMinor) {
      int nc = major * p.sizeMinor + minor;
      int n = nc / p.inSize.z;
      int c = nc - n * p.inSize.z;
      for (int loopX = 0, outX = outXBase; loopX < p.loopX & outX < p.outSize.x;
           loopX++, outX += blockDim.y) {
        // Setup X receptive field.
        int midX = outX * p.down.x + p.up.x - 1 - p.pad0.x;
        int inX = min(max(floor_div(midX, p.up.x), 0), p.inSize.x);
        int w =
            min(max(floor_div(midX + p.filterSize.x, p.up.x), 0), p.inSize.x) -
            inX;
        int filterX = midX + p.filterSize.x - (inX + 1) * p.up.x;
        if (p.flip) filterX = p.filterSize.x - 1 - filterX;

        // Initialize pointers.
        const T *xp =
            &((const T *)p.x)[inX * p.inStride.x + inY * p.inStride.y +
                              c * p.inStride.z + n * p.inStride.w];
        const float *fp =
            &p.f[filterX * p.filterStride.x + filterY * p.filterStride.y];
        int filterStepX = ((p.flip) ? p.up.x : -p.up.x) * p.filterStride.x;
        int filterStepY = ((p.flip) ? p.up.y : -p.up.y) * p.filterStride.y;

        // Inner loop.
        scalar_t v = 0;
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < w; x++) {
            v += (scalar_t)(*xp) * (scalar_t)(*fp);
            xp += p.inStride.x;
            fp += filterStepX;
          }
          xp += p.inStride.y - w * p.inStride.x;
          fp += filterStepY - w * filterStepX;
        }

        // Store result.
        v *= p.gain;
        ((T *)p.y)[outX * p.outStride.x + outY * p.outStride.y +
                   c * p.outStride.z + n * p.outStride.w] = (T)v;
      }
    }
}

//------------------------------------------------------------------------
// Specialized MUSA implementation for small filters.

template <class T, int upx, int upy, int downx, int downy, int filterW,
          int filterH, int tileOutW, int tileOutH, int loopMinor>
static __global__ void upfirdn2d_kernel_small(upfirdn2d_kernel_params p) {
  typedef typename InternalType<T>::scalar_t scalar_t;
  const int tileInW = ((tileOutW - 1) * downx + filterW - 1) / upx + 1;
  const int tileInH = ((tileOutH - 1) * downy + filterH - 1) / upy + 1;
  __shared__ volatile scalar_t sf[filterH][filterW];
  __shared__ volatile scalar_t sx[tileInH][tileInW][loopMinor];

  // Calculate tile index.
  int minorBase = blockIdx.x;
  int tileOutY = minorBase / p.launchMinor;
  minorBase -= tileOutY * p.launchMinor;
  minorBase *= loopMinor;
  tileOutY *= tileOutH;
  int tileOutXBase = blockIdx.y * p.loopX * tileOutW;
  int majorBase = blockIdx.z * p.loopMajor;
  if (tileOutXBase >= p.outSize.x | tileOutY >= p.outSize.y |
      majorBase >= p.sizeMajor)
    return;

  // Load filter (flipped).
  for (int tapIdx = threadIdx.x; tapIdx < filterH * filterW;
       tapIdx += blockDim.x) {
    int fy = tapIdx / filterW;
    int fx = tapIdx - fy * filterW;
    scalar_t v = 0;
    if (fx < p.filterSize.x & fy < p.filterSize.y) {
      int ffx = (p.flip) ? fx : p.filterSize.x - 1 - fx;
      int ffy = (p.flip) ? fy : p.filterSize.y - 1 - fy;
      v = (scalar_t)p.f[ffx * p.filterStride.x + ffy * p.filterStride.y];
    }
    sf[fy][fx] = v;
  }

  // Loop over major and X.
  for (int majorIdx = 0, major = majorBase;
       majorIdx < p.loopMajor & major < p.sizeMajor; majorIdx++, major++) {
    int baseNC = major * p.sizeMinor + minorBase;
    int n = baseNC / p.inSize.z;
    int baseC = baseNC - n * p.inSize.z;
    for (int loopX = 0, tileOutX = tileOutXBase;
         loopX < p.loopX & tileOutX < p.outSize.x;
         loopX++, tileOutX += tileOutW) {
      // Load input pixels.
      int tileMidX = tileOutX * downx + upx - 1 - p.pad0.x;
      int tileMidY = tileOutY * downy + upy - 1 - p.pad0.y;
      int tileInX = floor_div(tileMidX, upx);
      int tileInY = floor_div(tileMidY, upy);
      __syncthreads();
      for (int inIdx = threadIdx.x; inIdx < tileInH * tileInW * loopMinor;
           inIdx += blockDim.x) {
        int relC = inIdx;
        int relInX = relC / loopMinor;
        int relInY = relInX / tileInW;
        relC -= relInX * loopMinor;
        relInX -= relInY * tileInW;
        int c = baseC + relC;
        int inX = tileInX + relInX;
        int inY = tileInY + relInY;
        scalar_t v = 0;
        if (inX >= 0 & inY >= 0 & inX < p.inSize.x & inY < p.inSize.y &
            c < p.inSize.z)
          v = (scalar_t)(
              (const T *)p.x)[inX * p.inStride.x + inY * p.inStride.y +
                              c * p.inStride.z + n * p.inStride.w];
        sx[relInY][relInX][relC] = v;
      }

      // Loop over output pixels.
      __syncthreads();
      for (int outIdx = threadIdx.x; outIdx < tileOutH * tileOutW * loopMinor;
           outIdx += blockDim.x) {
        int relC = outIdx;
        int relOutX = relC / loopMinor;
        int relOutY = relOutX / tileOutW;
        relC -= relOutX * loopMinor;
        relOutX -= relOutY * tileOutW;
        int c = baseC + relC;
        int outX = tileOutX + relOutX;
        int outY = tileOutY + relOutY;

        // Setup receptive field.
        int midX = tileMidX + relOutX * downx;
        int midY = tileMidY + relOutY * downy;
        int inX = floor_div(midX, upx);
        int inY = floor_div(midY, upy);
        int relInX = inX - tileInX;
        int relInY = inY - tileInY;
        int filterX = (inX + 1) * upx - midX - 1;  // flipped
        int filterY = (inY + 1) * upy - midY - 1;  // flipped

        // Inner loop.
        if (outX < p.outSize.x & outY < p.outSize.y & c < p.outSize.z) {
          scalar_t v = 0;
#pragma unroll
          for (int y = 0; y < filterH / upy; y++)
#pragma unroll
            for (int x = 0; x < filterW / upx; x++)
              v += sx[relInY + y][relInX + x][relC] *
                   sf[filterY + y * upy][filterX + x * upx];
          v *= p.gain;
          ((T *)p.y)[outX * p.outStride.x + outY * p.outStride.y +
                     c * p.outStride.z + n * p.outStride.w] = (T)v;
        }
      }
    }
  }
}

//------------------------------------------------------------------------
// MUSA kernel selection.

template <class T>
upfirdn2d_kernel_spec choose_upfirdn2d_kernel(
    const upfirdn2d_kernel_params &p) {
  int s = p.inStride.z, fx = p.filterSize.x, fy = p.filterSize.y;
  upfirdn2d_kernel_spec spec = {(void *)upfirdn2d_kernel_large<T>, -1, -1, 1,
                                4};  // contiguous
  if (s == 1)
    spec = {(void *)upfirdn2d_kernel_large<T>, -1, -1, 4, 1};  // channels_last

  // No up/downsampling.
  if (p.up.x == 1 && p.up.y == 1 && p.down.x == 1 && p.down.y == 1) {
    // contiguous
    if (s != 1 && fx <= 24 && fy <= 24)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 24, 24, 64, 32, 1>,
              64, 32, 1, 1};
    if (s != 1 && fx <= 16 && fy <= 16)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 16, 16, 64, 32, 1>,
              64, 32, 1, 1};
    if (s != 1 && fx <= 7 && fy <= 7)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 7, 7, 64, 16, 1>,
              64, 16, 1, 1};
    if (s != 1 && fx <= 6 && fy <= 6)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 6, 6, 64, 16, 1>,
              64, 16, 1, 1};
    if (s != 1 && fx <= 5 && fy <= 5)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 5, 5, 64, 16, 1>,
              64, 16, 1, 1};
    if (s != 1 && fx <= 4 && fy <= 4)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 4, 4, 64, 16, 1>,
              64, 16, 1, 1};
    if (s != 1 && fx <= 3 && fy <= 3)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 3, 3, 64, 16, 1>,
              64, 16, 1, 1};
    if (s != 1 && fx <= 24 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 24, 1, 128, 8, 1>,
              128, 8, 1, 1};
    if (s != 1 && fx <= 16 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 16, 1, 128, 8, 1>,
              128, 8, 1, 1};
    if (s != 1 && fx <= 8 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 8, 1, 128, 8, 1>,
              128, 8, 1, 1};
    if (s != 1 && fx <= 1 && fy <= 24)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 1, 24, 32, 32, 1>,
              32, 32, 1, 1};
    if (s != 1 && fx <= 1 && fy <= 16)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 1, 16, 32, 32, 1>,
              32, 32, 1, 1};
    if (s != 1 && fx <= 1 && fy <= 8)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 1, 8, 32, 32, 1>,
              32, 32, 1, 1};
    // channels_last
    if (s == 1 && fx <= 24 && fy <= 24)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 24, 24, 32, 32, 1>,
              32, 32, 1, 1};
    if (s == 1 && fx <= 16 && fy <= 16)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 16, 16, 32, 32, 1>,
              32, 32, 1, 1};
    if (s == 1 && fx <= 7 && fy <= 7)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 7, 7, 16, 16, 8>,
              16, 16, 8, 1};
    if (s == 1 && fx <= 6 && fy <= 6)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 6, 6, 16, 16, 8>,
              16, 16, 8, 1};
    if (s == 1 && fx <= 5 && fy <= 5)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 5, 5, 16, 16, 8>,
              16, 16, 8, 1};
    if (s == 1 && fx <= 4 && fy <= 4)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 4, 4, 16, 16, 8>,
              16, 16, 8, 1};
    if (s == 1 && fx <= 3 && fy <= 3)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 3, 3, 16, 16, 8>,
              16, 16, 8, 1};
    if (s == 1 && fx <= 24 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 24, 1, 128, 1, 16>,
              128, 1, 16, 1};
    if (s == 1 && fx <= 16 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 16, 1, 128, 1, 16>,
              128, 1, 16, 1};
    if (s == 1 && fx <= 8 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 8, 1, 128, 1, 16>,
              128, 1, 16, 1};
    if (s == 1 && fx <= 1 && fy <= 24)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 1, 24, 1, 128, 16>,
              1, 128, 16, 1};
    if (s == 1 && fx <= 1 && fy <= 16)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 1, 16, 1, 128, 16>,
              1, 128, 16, 1};
    if (s == 1 && fx <= 1 && fy <= 8)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 1, 1, 8, 1, 128, 16>,
              1, 128, 16, 1};
  }

  // 2x upsampling.
  if (p.up.x == 2 && p.up.y == 2 && p.down.x == 1 && p.down.y == 1) {
    // contiguous
    if (s != 1 && fx <= 24 && fy <= 24)
      spec = {(void *)upfirdn2d_kernel_small<T, 2, 2, 1, 1, 24, 24, 64, 32, 1>,
              64, 32, 1, 1};
    if (s != 1 && fx <= 16 && fy <= 16)
      spec = {(void *)upfirdn2d_kernel_small<T, 2, 2, 1, 1, 16, 16, 64, 32, 1>,
              64, 32, 1, 1};
    if (s != 1 && fx <= 8 && fy <= 8)
      spec = {(void *)upfirdn2d_kernel_small<T, 2, 2, 1, 1, 8, 8, 64, 16, 1>,
              64, 16, 1, 1};
    if (s != 1 && fx <= 6 && fy <= 6)
      spec = {(void *)upfirdn2d_kernel_small<T, 2, 2, 1, 1, 6, 6, 64, 16, 1>,
              64, 16, 1, 1};
    if (s != 1 && fx <= 4 && fy <= 4)
      spec = {(void *)upfirdn2d_kernel_small<T, 2, 2, 1, 1, 4, 4, 64, 16, 1>,
              64, 16, 1, 1};
    if (s != 1 && fx <= 2 && fy <= 2)
      spec = {(void *)upfirdn2d_kernel_small<T, 2, 2, 1, 1, 2, 2, 64, 16, 1>,
              64, 16, 1, 1};
    // channels_last
    if (s == 1 && fx <= 24 && fy <= 24)
      spec = {(void *)upfirdn2d_kernel_small<T, 2, 2, 1, 1, 24, 24, 32, 32, 1>,
              32, 32, 1, 1};
    if (s == 1 && fx <= 16 && fy <= 16)
      spec = {(void *)upfirdn2d_kernel_small<T, 2, 2, 1, 1, 16, 16, 32, 32, 1>,
              32, 32, 1, 1};
    if (s == 1 && fx <= 8 && fy <= 8)
      spec = {(void *)upfirdn2d_kernel_small<T, 2, 2, 1, 1, 8, 8, 16, 16, 8>,
              16, 16, 8, 1};
    if (s == 1 && fx <= 6 && fy <= 6)
      spec = {(void *)upfirdn2d_kernel_small<T, 2, 2, 1, 1, 6, 6, 16, 16, 8>,
              16, 16, 8, 1};
    if (s == 1 && fx <= 4 && fy <= 4)
      spec = {(void *)upfirdn2d_kernel_small<T, 2, 2, 1, 1, 4, 4, 16, 16, 8>,
              16, 16, 8, 1};
    if (s == 1 && fx <= 2 && fy <= 2)
      spec = {(void *)upfirdn2d_kernel_small<T, 2, 2, 1, 1, 2, 2, 16, 16, 8>,
              16, 16, 8, 1};
  }
  if (p.up.x == 2 && p.up.y == 1 && p.down.x == 1 && p.down.y == 1) {
    // contiguous
    if (s != 1 && fx <= 24 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 2, 1, 1, 1, 24, 1, 128, 8, 1>,
              128, 8, 1, 1};
    if (s != 1 && fx <= 16 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 2, 1, 1, 1, 16, 1, 128, 8, 1>,
              128, 8, 1, 1};
    if (s != 1 && fx <= 8 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 2, 1, 1, 1, 8, 1, 128, 8, 1>,
              128, 8, 1, 1};
    // channels_last
    if (s == 1 && fx <= 24 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 2, 1, 1, 1, 24, 1, 128, 1, 16>,
              128, 1, 16, 1};
    if (s == 1 && fx <= 16 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 2, 1, 1, 1, 16, 1, 128, 1, 16>,
              128, 1, 16, 1};
    if (s == 1 && fx <= 8 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 2, 1, 1, 1, 8, 1, 128, 1, 16>,
              128, 1, 16, 1};
  }
  if (p.up.x == 1 && p.up.y == 2 && p.down.x == 1 && p.down.y == 1) {
    // contiguous
    if (s != 1 && fx <= 1 && fy <= 24)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 2, 1, 1, 1, 24, 32, 32, 1>,
              32, 32, 1, 1};
    if (s != 1 && fx <= 1 && fy <= 16)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 2, 1, 1, 1, 16, 32, 32, 1>,
              32, 32, 1, 1};
    if (s != 1 && fx <= 1 && fy <= 8)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 2, 1, 1, 1, 8, 32, 32, 1>,
              32, 32, 1, 1};
    // channels_last
    if (s == 1 && fx <= 1 && fy <= 24)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 2, 1, 1, 1, 24, 1, 128, 16>,
              1, 128, 16, 1};
    if (s == 1 && fx <= 1 && fy <= 16)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 2, 1, 1, 1, 16, 1, 128, 16>,
              1, 128, 16, 1};
    if (s == 1 && fx <= 1 && fy <= 8)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 2, 1, 1, 1, 8, 1, 128, 16>,
              1, 128, 16, 1};
  }

  // 2x downsampling.
  if (p.up.x == 1 && p.up.y == 1 && p.down.x == 2 && p.down.y == 2) {
    // contiguous
    if (s != 1 && fx <= 24 && fy <= 24)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 2, 2, 24, 24, 32, 16, 1>,
              32, 16, 1, 1};
    if (s != 1 && fx <= 16 && fy <= 16)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 2, 2, 16, 16, 32, 16, 1>,
              32, 16, 1, 1};
    if (s != 1 && fx <= 8 && fy <= 8)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 2, 2, 8, 8, 32, 8, 1>, 32,
              8, 1, 1};
    if (s != 1 && fx <= 6 && fy <= 6)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 2, 2, 6, 6, 32, 8, 1>, 32,
              8, 1, 1};
    if (s != 1 && fx <= 4 && fy <= 4)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 2, 2, 4, 4, 32, 8, 1>, 32,
              8, 1, 1};
    if (s != 1 && fx <= 2 && fy <= 2)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 2, 2, 2, 2, 32, 8, 1>, 32,
              8, 1, 1};
    // channels_last
    if (s == 1 && fx <= 24 && fy <= 24)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 2, 2, 24, 24, 16, 16, 1>,
              16, 16, 1, 1};
    if (s == 1 && fx <= 16 && fy <= 16)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 2, 2, 16, 16, 16, 16, 1>,
              16, 16, 1, 1};
    if (s == 1 && fx <= 8 && fy <= 8)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 2, 2, 8, 8, 8, 8, 8>, 8,
              8, 8, 1};
    if (s == 1 && fx <= 6 && fy <= 6)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 2, 2, 6, 6, 8, 8, 8>, 8,
              8, 8, 1};
    if (s == 1 && fx <= 4 && fy <= 4)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 2, 2, 4, 4, 8, 8, 8>, 8,
              8, 8, 1};
    if (s == 1 && fx <= 2 && fy <= 2)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 2, 2, 2, 2, 8, 8, 8>, 8,
              8, 8, 1};
  }
  if (p.up.x == 1 && p.up.y == 1 && p.down.x == 2 && p.down.y == 1) {
    // contiguous
    if (s != 1 && fx <= 24 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 2, 1, 24, 1, 64, 8, 1>,
              64, 8, 1, 1};
    if (s != 1 && fx <= 16 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 2, 1, 16, 1, 64, 8, 1>,
              64, 8, 1, 1};
    if (s != 1 && fx <= 8 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 2, 1, 8, 1, 64, 8, 1>, 64,
              8, 1, 1};
    // channels_last
    if (s == 1 && fx <= 24 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 2, 1, 24, 1, 64, 1, 8>,
              64, 1, 8, 1};
    if (s == 1 && fx <= 16 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 2, 1, 16, 1, 64, 1, 8>,
              64, 1, 8, 1};
    if (s == 1 && fx <= 8 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 2, 1, 8, 1, 64, 1, 8>, 64,
              1, 8, 1};
  }
  if (p.up.x == 1 && p.up.y == 1 && p.down.x == 1 && p.down.y == 2) {
    // contiguous
    if (s != 1 && fx <= 1 && fy <= 24)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 2, 1, 24, 32, 16, 1>,
              32, 16, 1, 1};
    if (s != 1 && fx <= 1 && fy <= 16)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 2, 1, 16, 32, 16, 1>,
              32, 16, 1, 1};
    if (s != 1 && fx <= 1 && fy <= 8)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 2, 1, 8, 32, 16, 1>,
              32, 16, 1, 1};
    // channels_last
    if (s == 1 && fx <= 1 && fy <= 24)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 2, 1, 24, 1, 64, 8>, 1,
              64, 8, 1};
    if (s == 1 && fx <= 1 && fy <= 16)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 2, 1, 16, 1, 64, 8>, 1,
              64, 8, 1};
    if (s == 1 && fx <= 1 && fy <= 8)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 2, 1, 8, 1, 64, 8>, 1,
              64, 8, 1};
  }

  // 4x upsampling.
  if (p.up.x == 4 && p.up.y == 4 && p.down.x == 1 && p.down.y == 1) {
    // contiguous
    if (s != 1 && fx <= 48 && fy <= 48)
      spec = {(void *)upfirdn2d_kernel_small<T, 4, 4, 1, 1, 48, 48, 64, 32, 1>,
              64, 32, 1, 1};
    if (s != 1 && fx <= 32 && fy <= 32)
      spec = {(void *)upfirdn2d_kernel_small<T, 4, 4, 1, 1, 32, 32, 64, 32, 1>,
              64, 32, 1, 1};
    // channels_last
    if (s == 1 && fx <= 48 && fy <= 48)
      spec = {(void *)upfirdn2d_kernel_small<T, 4, 4, 1, 1, 48, 48, 32, 32, 1>,
              32, 32, 1, 1};
    if (s == 1 && fx <= 32 && fy <= 32)
      spec = {(void *)upfirdn2d_kernel_small<T, 4, 4, 1, 1, 32, 32, 32, 32, 1>,
              32, 32, 1, 1};
  }
  if (p.up.x == 4 && p.up.y == 1 && p.down.x == 1 && p.down.y == 1) {
    // contiguous
    if (s != 1 && fx <= 48 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 4, 1, 1, 1, 48, 1, 128, 8, 1>,
              128, 8, 1, 1};
    if (s != 1 && fx <= 32 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 4, 1, 1, 1, 32, 1, 128, 8, 1>,
              128, 8, 1, 1};
    // channels_last
    if (s == 1 && fx <= 48 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 4, 1, 1, 1, 48, 1, 128, 1, 16>,
              128, 1, 16, 1};
    if (s == 1 && fx <= 32 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 4, 1, 1, 1, 32, 1, 128, 1, 16>,
              128, 1, 16, 1};
  }
  if (p.up.x == 1 && p.up.y == 4 && p.down.x == 1 && p.down.y == 1) {
    // contiguous
    if (s != 1 && fx <= 1 && fy <= 48)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 4, 1, 1, 1, 48, 32, 32, 1>,
              32, 32, 1, 1};
    if (s != 1 && fx <= 1 && fy <= 32)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 4, 1, 1, 1, 32, 32, 32, 1>,
              32, 32, 1, 1};
    // channels_last
    if (s == 1 && fx <= 1 && fy <= 48)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 4, 1, 1, 1, 48, 1, 128, 16>,
              1, 128, 16, 1};
    if (s == 1 && fx <= 1 && fy <= 32)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 4, 1, 1, 1, 32, 1, 128, 16>,
              1, 128, 16, 1};
  }

  // 4x downsampling (inefficient).
  if (p.up.x == 1 && p.up.y == 1 && p.down.x == 4 && p.down.y == 1) {
    // contiguous
    if (s != 1 && fx <= 48 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 4, 1, 48, 1, 32, 8, 1>,
              32, 8, 1, 1};
    if (s != 1 && fx <= 32 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 4, 1, 32, 1, 32, 8, 1>,
              32, 8, 1, 1};
    // channels_last
    if (s == 1 && fx <= 48 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 4, 1, 48, 1, 32, 1, 8>,
              32, 1, 8, 1};
    if (s == 1 && fx <= 32 && fy <= 1)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 4, 1, 32, 1, 32, 1, 8>,
              32, 1, 8, 1};
  }
  if (p.up.x == 1 && p.up.y == 1 && p.down.x == 1 && p.down.y == 4) {
    // contiguous
    if (s != 1 && fx <= 1 && fy <= 48)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 4, 1, 48, 32, 8, 1>,
              32, 8, 1, 1};
    if (s != 1 && fx <= 1 && fy <= 32)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 4, 1, 32, 32, 8, 1>,
              32, 8, 1, 1};
    // channels_last
    if (s == 1 && fx <= 1 && fy <= 48)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 4, 1, 48, 1, 32, 8>, 1,
              32, 8, 1};
    if (s == 1 && fx <= 1 && fy <= 32)
      spec = {(void *)upfirdn2d_kernel_small<T, 1, 1, 1, 4, 1, 32, 1, 32, 8>, 1,
              32, 8, 1};
  }
  return spec;
}

//------------------------------------------------------------------------
// Template specializations.

template upfirdn2d_kernel_spec choose_upfirdn2d_kernel<double>(
    const upfirdn2d_kernel_params &p);
template upfirdn2d_kernel_spec choose_upfirdn2d_kernel<float>(
    const upfirdn2d_kernel_params &p);
template upfirdn2d_kernel_spec choose_upfirdn2d_kernel<c10::Half>(
    const upfirdn2d_kernel_params &p);

//------------------------------------------------------------------------

//------------------------------------------------------------------------

torch::Tensor upfirdn2d_op(torch::Tensor x, torch::Tensor f, int upx, int upy,
                           int downx, int downy, int padx0, int padx1,
                           int pady0, int pady1, bool flip, float gain) {
  // Validate arguments.
  TORCH_CHECK(x.is_privateuseone(), "x must reside on MUSA device");
  TORCH_CHECK(f.device() == x.device(),
              "f must reside on the same device as x");
  TORCH_CHECK(f.dtype() == torch::kFloat, "f must be float32");
  TORCH_CHECK(x.numel() <= INT_MAX, "x is too large");
  TORCH_CHECK(f.numel() <= INT_MAX, "f is too large");
  TORCH_CHECK(x.numel() > 0, "x has zero size");
  TORCH_CHECK(f.numel() > 0, "f has zero size");
  TORCH_CHECK(x.dim() == 4, "x must be rank 4");
  TORCH_CHECK(f.dim() == 2, "f must be rank 2");
  TORCH_CHECK((x.size(0) - 1) * x.stride(0) + (x.size(1) - 1) * x.stride(1) +
                      (x.size(2) - 1) * x.stride(2) +
                      (x.size(3) - 1) * x.stride(3) <=
                  INT_MAX,
              "x memory footprint is too large");
  TORCH_CHECK(f.size(0) >= 1 && f.size(1) >= 1, "f must be at least 1x1");
  TORCH_CHECK(upx >= 1 && upy >= 1, "upsampling factor must be at least 1");
  TORCH_CHECK(downx >= 1 && downy >= 1,
              "downsampling factor must be at least 1");

  // Create output tensor.
  const at::musa::OptionalMUSAGuard device_guard(device_of(x));
  int outW =
      ((int)x.size(3) * upx + padx0 + padx1 - (int)f.size(1) + downx) / downx;
  int outH =
      ((int)x.size(2) * upy + pady0 + pady1 - (int)f.size(0) + downy) / downy;
  TORCH_CHECK(outW >= 1 && outH >= 1, "output must be at least 1x1");
  torch::Tensor y = torch::empty({x.size(0), x.size(1), outH, outW},
                                 x.options(), x.suggest_memory_format());
  TORCH_CHECK(y.numel() <= INT_MAX, "output is too large");
  TORCH_CHECK((y.size(0) - 1) * y.stride(0) + (y.size(1) - 1) * y.stride(1) +
                      (y.size(2) - 1) * y.stride(2) +
                      (y.size(3) - 1) * y.stride(3) <=
                  INT_MAX,
              "output memory footprint is too large");

  // Initialize MUSA kernel parameters.
  upfirdn2d_kernel_params p;
  p.x = x.data_ptr();
  p.f = f.data_ptr<float>();
  p.y = y.data_ptr();
  p.up = make_int2(upx, upy);
  p.down = make_int2(downx, downy);
  p.pad0 = make_int2(padx0, pady0);
  p.flip = (flip) ? 1 : 0;
  p.gain = gain;
  p.inSize =
      make_int4((int)x.size(3), (int)x.size(2), (int)x.size(1), (int)x.size(0));
  p.inStride = make_int4((int)x.stride(3), (int)x.stride(2), (int)x.stride(1),
                         (int)x.stride(0));
  p.filterSize = make_int2((int)f.size(1), (int)f.size(0));
  p.filterStride = make_int2((int)f.stride(1), (int)f.stride(0));
  p.outSize =
      make_int4((int)y.size(3), (int)y.size(2), (int)y.size(1), (int)y.size(0));
  p.outStride = make_int4((int)y.stride(3), (int)y.stride(2), (int)y.stride(1),
                          (int)y.stride(0));
  p.sizeMajor = (p.inStride.z == 1) ? p.inSize.w : p.inSize.w * p.inSize.z;
  p.sizeMinor = (p.inStride.z == 1) ? p.inSize.z : 1;

  // Choose MUSA kernel.
  upfirdn2d_kernel_spec spec;
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "upfirdn2d_musa", [&] {
    spec = choose_upfirdn2d_kernel<scalar_t>(p);
  });

  // Set looping options.
  p.loopMajor = (p.sizeMajor - 1) / 16384 + 1;
  p.loopMinor = spec.loopMinor;
  p.loopX = spec.loopX;
  p.launchMinor = (p.sizeMinor - 1) / p.loopMinor + 1;
  p.launchMajor = (p.sizeMajor - 1) / p.loopMajor + 1;

  // Compute grid size.
  dim3 blockSize, gridSize;
  if (spec.tileOutW < 0)  // large
  {
    blockSize = dim3(4, 32, 1);
    gridSize =
        dim3(((p.outSize.y - 1) / blockSize.x + 1) * p.launchMinor,
             (p.outSize.x - 1) / (blockSize.y * p.loopX) + 1, p.launchMajor);
  } else  // small
  {
    blockSize = dim3(256, 1, 1);
    gridSize =
        dim3(((p.outSize.y - 1) / spec.tileOutH + 1) * p.launchMinor,
             (p.outSize.x - 1) / (spec.tileOutW * p.loopX) + 1, p.launchMajor);
  }

  // Launch MUSA kernel.
  void *args[] = {&p};
#ifdef MMCV_WITH_HIP
  AT_MUSA_CHECK(hipLaunchKernel(spec.kernel, gridSize, blockSize, args, 0,
                                c10::musa::getCurrentMUSAStream()));
#else
  AT_MUSA_CHECK(musaLaunchKernel(spec.kernel, gridSize, blockSize, args, 0,
                                 c10::musa::getCurrentMUSAStream()));
#endif

  return y;
}
#else
#warning "upfirdn2d is supported when MUSA_ARCH > 21"
#endif  //MUSA_ARCH
