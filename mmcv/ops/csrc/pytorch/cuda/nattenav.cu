/*!
**************************************************************************************************
* NATTEN-AV TORCH EXTENSION (CUDA)
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from
*https://github.com/SHI-Labs/Neighborhood-Attention-Transformer
**************************************************************************************************
*/
#include <ATen/AccumulateType.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <ATen/native/cuda/KernelUtils.cuh>
#include <vector>

#include "nattenav_cuda_kernel.cuh"
#include "pytorch_cpp_helper.hpp"
#include "pytorch_cuda_helper.hpp"

/*
Beware: here be dragons! Edit with caution.
This file is not well documented yet. We're working on that.
That said, we welcome issues, emails, and of course PRs.
There's now a bunch of kernels, some special purpose, some general purpose.
Special purpose ones use shared memory and are highly sensitive to edits in
their current form. There's also a specific FP16 kernel for everything now.
*/

#define WARP_SIZE 32

#define KERNEL_SIZE_13 13
#define KERNEL_SIZE_11 11
#define KERNEL_SIZE_9 9
#define KERNEL_SIZE_7 7
#define KERNEL_SIZE_5 5
#define NEIGHBORHOOD_SIZE_13 6
#define NEIGHBORHOOD_SIZE_11 5
#define NEIGHBORHOOD_SIZE_9 4
#define NEIGHBORHOOD_SIZE_7 3
#define NEIGHBORHOOD_SIZE_5 2
// Always keep batchthreads 1, because we want each thread block to process one
// 1 sample 1 head
#define BATCHTHREADS_13 1
#define BATCHTHREADS_11 1
#define BATCHTHREADS_9 1
#define BATCHTHREADS_7 1
#define BATCHTHREADS_5 1
// Tile is the number of pixels across each axis that are processed within a
// single threadblock So far the best tile size for Kernel size 7 is 3x3.
#define TILE_9 3
#define TILE_7 3
#define TILE_5 4

#define TILE_11_X 2
#define TILE_11_Y 3
#define TILE_13_X 2
#define TILE_13_Y 3
// Each of the 3x3 pixels has 7x7 key neighbors in this case, therefore the tile
// size for keys will 7 + 3 - 1 = 9x9
#define KTILE_9 11
#define KTILE_7 9
#define KTILE_5 8

#define KTILE_11_X 12
#define KTILE_11_Y 13
#define KTILE_13_X 14
#define KTILE_13_Y 15
// 7x7 kernel, and we want each threadblock to process the entire neighborhood
// for each QUERY in its tile, so we'll have 7x7 * 3x3 = 21x21 Also keep in mind
// these 21 threads are across each axis, so it's 21x21 threads total 21x21 =
// 441 < 1024 Ensure it's less than 1024, which is the max number of threads per
// threadblock
#define XYTHREADS_9 27
#define XYTHREADS_7 21
#define XYTHREADS_5 20

#define XTHREADS_11 33
#define YTHREADS_11 22
#define XTHREADS_13 39
#define YTHREADS_13 26

// DIM is fixed at 32 for now
#define DIM_32 32
#define DIMHALF_32 16  // FP16 stored in half2 => half the dims
// There's 32 * 3x3 QUERY cells to store, and 32 * 10x10 KEY cells
// The former is 288 < 441 threads, so each thread can copy over one QUERY cell
// exactly, and we'll have empty threads too But that's not the case for the
// latter, which is 3200 and it's not < 441 But we can have each thread load
// more cells instead. 8 is optimal since it will maximize utility So copy 8
// dims per KEY pixel in each thread
#define KITERS_32 8
#define KHALFITERS_32 4  // FP16 stored in half2 => half the dims
// and DIM = 32 / 8 = 4, hence 4 is the stride.
#define KSTRIDE_32 4
// For kernel size 5, we have to do 2 query dims per thread, because we have
// fewer threads in each threadblock than the total number of queries.
#define QITERS_5 2
#define QSTRIDE_5 16

// This is just for the other kernels that are not using SMEM
#define CUDA_NUM_THREADS 1024
#define CUDA_NUM_THREADS_F 512
#define CUDA_NUM_THREADS_FP16 512
#define CUDA_NUM_THREADS_V 512
#define CUDA_NUM_THREADS_V16 256

#define AT_DISPATCH_HALF_TYPES(SCALARTYPE1, TYPE, NAME, ...)                \
  [&] {                                                                     \
    const auto &the_type = TYPE;                                            \
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

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, typename scalar_t>
__global__ void nattenav_cuda_forward_kernel_fp16(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        attn,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        value,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits> out,
    const int height, const int width, const int heads, const int dimhalf,
    const int totalElements) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearIndex < totalElements) {
    __half2 *value2 = reinterpret_cast<__half2 *>(value.data());
    __half2 *out2 = reinterpret_cast<__half2 *>(out.data());
    int indtmp1 = linearIndex / dimhalf;
    const int d = linearIndex - indtmp1 * dimhalf;
    int indtmp2 = indtmp1 / width;
    const int j = indtmp1 - indtmp2 * width;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1 / height;
    const int i = indtmp1 - indtmp2 * height;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1 / heads;
    const int h = indtmp1 - indtmp2 * heads;
    const int b = indtmp2;

    const int ni = get_window_start(i, height, NEIGHBORHOOD_SIZE);
    const int nj = get_window_start(j, width, NEIGHBORHOOD_SIZE);
    __half2 updt = __float2half2_rn(0.f);
    int attnOffset = b * attn.stride(0) + h * attn.stride(1) +
                     i * attn.stride(2) + j * attn.stride(3);
    const int stride2 = dimhalf * width;
    const int valueOffset =
        b * (stride2 * height * heads) + h * (stride2 * height) + d;
#pragma unroll
    for (int xi = ni; xi < ni + KERNEL_SIZE; ++xi)
#pragma unroll
      for (int xj = nj; xj < nj + KERNEL_SIZE; ++xj) {
        const int valueIndex = valueOffset + xi * stride2 + xj * dimhalf;
        scalar_t a = attn.data()[attnOffset];
        updt = __hfma2(__halves2half2(a, a), value2[valueIndex], updt);
        ++attnOffset;
      }
    out2[linearIndex] = updt;
  }
}

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, typename scalar_t>
__global__ void nattenav_cuda_forward_kernel_fp32(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        attn,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        value,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits> out,
    const int height, const int width, const int heads, const int dim,
    const int totalElements) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearIndex < totalElements) {
    int indtmp1 = linearIndex / dim;
    const int d = linearIndex - indtmp1 * dim;
    int indtmp2 = indtmp1 / width;
    const int j = indtmp1 - indtmp2 * width;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1 / height;
    const int i = indtmp1 - indtmp2 * height;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1 / heads;
    const int h = indtmp1 - indtmp2 * heads;
    const int b = indtmp2;

    const int ni = get_window_start(i, height, NEIGHBORHOOD_SIZE);
    const int nj = get_window_start(j, width, NEIGHBORHOOD_SIZE);
    scalar_t updt = scalar_t(0);
    int attnOffset = b * attn.stride(0) + h * attn.stride(1) +
                     i * attn.stride(2) + j * attn.stride(3);
    const int valueOffset = b * value.stride(0) + h * value.stride(1) + d;
#pragma unroll
    for (int xi = ni; xi < ni + KERNEL_SIZE; ++xi)
#pragma unroll
      for (int xj = nj; xj < nj + KERNEL_SIZE; ++xj) {
        const int valueIndex =
            valueOffset + xi * value.stride(2) + xj * value.stride(3);
        updt += attn.data()[attnOffset] * value.data()[valueIndex];
        ++attnOffset;
      }
    out.data()[linearIndex] = updt;
  }
}

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, typename scalar_t>
__global__ void nattena_cuda_backward_kernel_fp16(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        d_out,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        value,
    const int height, const int width, const int batch_size, const int heads,
    const int dimhalf) {
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (z < batch_size * heads) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < height * width) {
      const int y = blockIdx.y * blockDim.y + threadIdx.y;
      if (y < KERNEL_SIZE * KERNEL_SIZE) {
        __half2 *d_out2 = reinterpret_cast<__half2 *>(d_out.data());
        __half2 *value2 = reinterpret_cast<__half2 *>(value.data());
        const int b = z / heads;
        const int h = z - b * heads;
        const int ki = y / KERNEL_SIZE;
        const int kj = y - ki * KERNEL_SIZE;
        const int i = x / width;
        const int j = x - i * width;
        const int ni = get_window_start(i, height, NEIGHBORHOOD_SIZE);
        const int nj = get_window_start(j, width, NEIGHBORHOOD_SIZE);
        __half2 updt = __float2half2_rn(0.f);
        const int stride2 = dimhalf * width;
        const int batchHeadOffset =
            b * (stride2 * height * heads) + h * (stride2 * height);
        const int d_outOffset = batchHeadOffset + i * stride2 + j * dimhalf;
        const int valueOffset =
            batchHeadOffset + (ki + ni) * stride2 + (kj + nj) * dimhalf;
#pragma unroll
        for (int dimOffset = 0; dimOffset < dimhalf; ++dimOffset)
          updt = __hfma2(d_out2[d_outOffset + dimOffset],
                         value2[valueOffset + dimOffset], updt);
        const int index = b * d_attn.stride(0) + h * d_attn.stride(1) +
                          i * d_attn.stride(2) + j * d_attn.stride(3) +
                          y * d_attn.stride(4);
        d_attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y));
      }
    }
  }
}

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, typename scalar_t>
__global__ void nattena_cuda_backward_kernel_fp32(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        d_out,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        value,
    const int height, const int width, const int batch_size, const int heads,
    const int dim) {
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (z < batch_size * heads) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < height * width) {
      const int y = blockIdx.y * blockDim.y + threadIdx.y;
      if (y < KERNEL_SIZE * KERNEL_SIZE) {
        const int b = z / heads;
        const int h = z - b * heads;
        const int ki = y / KERNEL_SIZE;
        const int kj = y - ki * KERNEL_SIZE;
        const int i = x / width;
        const int j = x - i * width;
        const int ni = get_window_start(i, height, NEIGHBORHOOD_SIZE);
        const int nj = get_window_start(j, width, NEIGHBORHOOD_SIZE);
        scalar_t updt = scalar_t(0);
        const int batchHeadOffset = b * d_out.stride(0) + h * d_out.stride(1);
        const int d_outOffset =
            batchHeadOffset + i * d_out.stride(2) + j * d_out.stride(3);
        const int valueOffset = batchHeadOffset + (ki + ni) * value.stride(2) +
                                (kj + nj) * value.stride(3);
#pragma unroll
        for (int dimOffset = 0; dimOffset < dim; ++dimOffset)
          updt += d_out.data()[d_outOffset + dimOffset] *
                  value.data()[valueOffset + dimOffset];
        const int index = b * d_attn.stride(0) + h * d_attn.stride(1) +
                          i * d_attn.stride(2) + j * d_attn.stride(3) +
                          y * d_attn.stride(4);
        d_attn.data()[index] = updt;
      }
    }
  }
}

/* TODO: FIX BANK CONFLICTS */
template <typename scalar_t>
__global__ void nattena_cuda_backward_kernel_fp16_5x5_32(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        d_out,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        value,
    const int height, const int width, const int batch_size, const int heads) {
  // Because batch heads have stride 1 per threadblock, we can just use blockIdx
  // since blockDim will be 1 and threadIdx will always be 0. const int z =
  // blockIdx.z * blockDim.z + threadIdx.z;
  const int z = blockIdx.z;
  const int b = z / heads;
  const int h = z - b * heads;
  // Not needed again because it will always be true.
  // if (z < batch_size * heads)
  // {
  const int sx = blockIdx.x * XYTHREADS_5;
  const int x = sx + threadIdx.x;
  const int sy = blockIdx.y * XYTHREADS_5;
  const int y = sy + threadIdx.y;
  const int lti = threadIdx.y * XYTHREADS_5 + threadIdx.x;
  const int stride2 = DIMHALF_32 * width;
  const int batchHeadOffset =
      b * (stride2 * height * heads) + h * (stride2 * height);
  const int si = sy / KERNEL_SIZE_5;
  const int sj = sx / KERNEL_SIZE_5;
  const int sni = get_window_start(si, height, NEIGHBORHOOD_SIZE_5);
  const int snj = get_window_start(sj, width, NEIGHBORHOOD_SIZE_5);
  __shared__ __half2 tile[TILE_5 * TILE_5][DIM_32 + 3];
  __shared__ __half2 kTile[KTILE_5 * KTILE_5][DIM_32 + 3];
  __half2 *d_out2 = reinterpret_cast<__half2 *>(d_out.data());
  __half2 *value2 = reinterpret_cast<__half2 *>(value.data());

  /* d_out tile */
  const int qtx = lti / DIMHALF_32;
  const int qty = lti - qtx * DIMHALF_32;
  if (qtx < TILE_5 * TILE_5) {
    int qi = qtx / TILE_5;
    const int qj = qtx - qi * TILE_5 + sj;
    qi += si;
    if (qi < height && qj < width) {
      tile[qtx][qty] =
          d_out2[batchHeadOffset + qi * stride2 + qj * DIMHALF_32 + qty];
    }
  }
  /* value tile */
  const int ktx = lti / KSTRIDE_32;
  const int kty = (lti - ktx * KSTRIDE_32) * KHALFITERS_32;
  if (ktx < KTILE_5 * KTILE_5) {
    int bi = ktx / KTILE_5;
    const int bj = ktx - bi * KTILE_5 + snj;
    bi += sni;
    if (bi < height && bj < width) {
      const int valueOffset =
          batchHeadOffset + bi * stride2 + bj * DIMHALF_32 + kty;
#pragma unroll
      for (int ti = 0; ti < KHALFITERS_32; ++ti)
        kTile[ktx][kty + ti] = value2[valueOffset + ti];
    }
  }
  __syncthreads();
  if (y < height * KERNEL_SIZE_5 && x < width * KERNEL_SIZE_5) {
    const int i = y / KERNEL_SIZE_5;
    const int ki = y - i * KERNEL_SIZE_5;
    const int j = x / KERNEL_SIZE_5;
    const int kj = x - j * KERNEL_SIZE_5;
    const int ni = get_window_start(i, height, NEIGHBORHOOD_SIZE_5);
    const int nj = get_window_start(j, width, NEIGHBORHOOD_SIZE_5);
    __half2 updt = __float2half2_rn(0.f);
    const int d_outIdx = (i - si) * TILE_5 + (j - sj);
    const int valueIdx = (ni + ki - sni) * KTILE_5 + (nj + kj - snj);

#pragma unroll
    for (int dimOffset = 0; dimOffset < DIMHALF_32; ++dimOffset)
      updt =
          __hfma2(tile[d_outIdx][dimOffset], kTile[valueIdx][dimOffset], updt);
    const int index = b * d_attn.stride(0) + h * d_attn.stride(1) +
                      i * d_attn.stride(2) + j * d_attn.stride(3) +
                      ki * KERNEL_SIZE_5 + kj;
    d_attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y));
  }
  //}
}

/* TODO: CHECK BANK CONFLICTS */
template <typename scalar_t>
__global__ void nattena_cuda_backward_kernel_fp32_5x5_32(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        d_out,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        value,
    const int height, const int width, const int batch_size, const int heads) {
  // Because batch heads have stride 1 per threadblock, we can just use blockIdx
  // since blockDim will be 1 and threadIdx will always be 0. const int z =
  // blockIdx.z * blockDim.z + threadIdx.z;
  const int z = blockIdx.z;
  const int b = z / heads;
  const int h = z - b * heads;
  // Not needed again because it will always be true.
  // if (z < batch_size * heads)
  // {
  const int sx = blockIdx.x * XYTHREADS_5;
  const int x = sx + threadIdx.x;
  const int sy = blockIdx.y * XYTHREADS_5;
  const int y = sy + threadIdx.y;
  const int lti = threadIdx.y * XYTHREADS_5 + threadIdx.x;
  const int batchHeadOffset = b * d_out.stride(0) + h * d_out.stride(1);
  const int si = sy / KERNEL_SIZE_5;
  const int sj = sx / KERNEL_SIZE_5;
  const int sni = get_window_start(si, height, NEIGHBORHOOD_SIZE_5);
  const int snj = get_window_start(sj, width, NEIGHBORHOOD_SIZE_5);
  __shared__ scalar_t tile[TILE_5 * TILE_5][DIM_32 + 3];
  __shared__ scalar_t kTile[KTILE_5 * KTILE_5][DIM_32 + 3];

  /* d_out tile */
  const int qtx = lti / QSTRIDE_5;
  const int qty = (lti - qtx * QSTRIDE_5) * QITERS_5;
  if (qtx < TILE_5 * TILE_5) {
    int qi = qtx / TILE_5;
    const int qj = qtx - qi * TILE_5 + sj;
    qi += si;
    if (qi < height && qj < width) {
#pragma unroll
      for (int ti = 0; ti < QITERS_5; ++ti)
        tile[qtx][qty + ti] =
            d_out.data()[batchHeadOffset + qi * d_out.stride(2) +
                         qj * d_out.stride(3) + qty + ti];
    }
  }
  /* value tile */
  const int ktx = lti / KSTRIDE_32;
  const int kty = (lti - ktx * KSTRIDE_32) * KITERS_32;
  if (ktx < KTILE_5 * KTILE_5) {
    int bi = ktx / KTILE_5;
    const int bj = ktx - bi * KTILE_5 + snj;
    bi += sni;
    if (bi < height && bj < width) {
      const int valueOffset =
          batchHeadOffset + bi * d_out.stride(2) + bj * d_out.stride(3) + kty;
#pragma unroll
      for (int ti = 0; ti < KITERS_32; ++ti)
        kTile[ktx][kty + ti] = value.data()[valueOffset + ti];
    }
  }
  __syncthreads();
  if (y < height * KERNEL_SIZE_5 && x < width * KERNEL_SIZE_5) {
    const int i = y / KERNEL_SIZE_5;
    const int ki = y - i * KERNEL_SIZE_5;
    const int j = x / KERNEL_SIZE_5;
    const int kj = x - j * KERNEL_SIZE_5;
    const int ni = get_window_start(i, height, NEIGHBORHOOD_SIZE_5);
    const int nj = get_window_start(j, width, NEIGHBORHOOD_SIZE_5);
    scalar_t updt = scalar_t(0);
    const int d_outIdx = (i - si) * TILE_5 + (j - sj);
    const int valueIdx = (ni + ki - sni) * KTILE_5 + (nj + kj - snj);

#pragma unroll
    for (int dimOffset = 0; dimOffset < DIM_32; ++dimOffset)
      updt += tile[d_outIdx][dimOffset] * kTile[valueIdx][dimOffset];

    const int index = b * d_attn.stride(0) + h * d_attn.stride(1) +
                      i * d_attn.stride(2) + j * d_attn.stride(3) +
                      ki * KERNEL_SIZE_5 + kj;
    d_attn.data()[index] = updt;
  }
  //}
}

template <int TILE, int KTILE, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE,
          int XYTHREADS, typename scalar_t>
__global__ void nattena_cuda_backward_kernel_fp16_7x7_9x9_32(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        d_out,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        value,
    const int height, const int width, const int batch_size, const int heads) {
  // Because batch heads have stride 1 per threadblock, we can just use blockIdx
  // since blockDim will be 1 and threadIdx will always be 0. const int z =
  // blockIdx.z * blockDim.z + threadIdx.z;
  const int z = blockIdx.z;
  const int b = z / heads;
  const int h = z - b * heads;
  // Not needed again because it will always be true.
  // if (z < batch_size * heads)
  // {
  const int sx = blockIdx.x * XYTHREADS;
  const int x = sx + threadIdx.x;
  const int sy = blockIdx.y * XYTHREADS;
  const int y = sy + threadIdx.y;
  const int lti = threadIdx.y * XYTHREADS + threadIdx.x;
  const int stride2 = DIMHALF_32 * width;
  const int batchHeadOffset =
      b * (stride2 * height * heads) + h * (stride2 * height);
  const int si = sy / KERNEL_SIZE;
  const int sj = sx / KERNEL_SIZE;
  const int sni = get_window_start(si, height, NEIGHBORHOOD_SIZE);
  const int snj = get_window_start(sj, width, NEIGHBORHOOD_SIZE);
  __shared__ __half2 tile[TILE * TILE][DIM_32 + 3];
  __shared__ __half2 kTile[KTILE * KTILE][DIM_32 + 3];
  __half2 *d_out2 = reinterpret_cast<__half2 *>(d_out.data());
  __half2 *value2 = reinterpret_cast<__half2 *>(value.data());

  /* d_out tile */
  const int qtx = lti / DIM_32;
  const int qtyp = lti - qtx * DIM_32;
  const int qdi = qtyp / KHALFITERS_32;
  const int qdj = qtyp - qdi * KHALFITERS_32;
  const int qty = qdi * KITERS_32 + qdj;
  if (qtx < TILE * TILE && qtyp < DIMHALF_32) {
    int qi = qtx / TILE;
    const int qj = qtx - qi * TILE + sj;
    qi += si;
    if (qi < height && qj < width)
      tile[qtx][qty] =
          d_out2[batchHeadOffset + qi * stride2 + qj * DIMHALF_32 + qtyp];
  }
  /* value tile */
  const int ktx = lti / KSTRIDE_32;
  const int kty = (lti - ktx * KSTRIDE_32) * KHALFITERS_32;
  if (ktx < KTILE * KTILE) {
    int bi = ktx / KTILE;
    const int bj = ktx - bi * KTILE + snj;
    bi += sni;
    if (bi < height && bj < width) {
      const int valueOffset =
          batchHeadOffset + bi * stride2 + bj * DIMHALF_32 + kty;
#pragma unroll
      for (int ti = 0; ti < KHALFITERS_32; ++ti)
        kTile[ktx][kty * 2 + ti] = value2[valueOffset + ti];
    }
  }
  __syncthreads();
  if (y < height * KERNEL_SIZE && x < width * KERNEL_SIZE) {
    const int i = y / KERNEL_SIZE;
    const int ki = y - i * KERNEL_SIZE;
    const int j = x / KERNEL_SIZE;
    const int kj = x - j * KERNEL_SIZE;
    const int ni = get_window_start(i, height, NEIGHBORHOOD_SIZE);
    const int nj = get_window_start(j, width, NEIGHBORHOOD_SIZE);
    __half2 updt = __float2half2_rn(0.f);
    const int d_outIdx = (i - si) * TILE + (j - sj);
    const int valueIdx = (ni + ki - sni) * KTILE + (nj + kj - snj);

#pragma unroll
    for (int di = 0; di < KSTRIDE_32; ++di)
#pragma unroll
      for (int dj = 0; dj < KHALFITERS_32; ++dj)
        updt = __hfma2(tile[d_outIdx][di * KITERS_32 + dj],
                       kTile[valueIdx][di * KITERS_32 + dj], updt);
    const int index = b * d_attn.stride(0) + h * d_attn.stride(1) +
                      i * d_attn.stride(2) + j * d_attn.stride(3) +
                      ki * KERNEL_SIZE + kj;
    d_attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y));
  }
  //}
}

template <int TILE, int KTILE, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE,
          int XYTHREADS, typename scalar_t>
__global__ void nattena_cuda_backward_kernel_fp32_7x7_9x9_32(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        d_out,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        value,
    const int height, const int width, const int batch_size, const int heads) {
  // Because batch heads have stride 1 per threadblock, we can just use blockIdx
  // since blockDim will be 1 and threadIdx will always be 0. const int z =
  // blockIdx.z * blockDim.z + threadIdx.z;
  const int z = blockIdx.z;
  const int b = z / heads;
  const int h = z - b * heads;
  // Not needed again because it will always be true.
  // if (z < batch_size * heads)
  // {
  const int sx = blockIdx.x * XYTHREADS;
  const int x = sx + threadIdx.x;
  const int sy = blockIdx.y * XYTHREADS;
  const int y = sy + threadIdx.y;
  const int lti = threadIdx.y * XYTHREADS + threadIdx.x;
  const int batchHeadOffset = b * d_out.stride(0) + h * d_out.stride(1);
  const int si = sy / KERNEL_SIZE;
  const int sj = sx / KERNEL_SIZE;
  const int sni = get_window_start(si, height, NEIGHBORHOOD_SIZE);
  const int snj = get_window_start(sj, width, NEIGHBORHOOD_SIZE);
  __shared__ scalar_t tile[TILE * TILE][DIM_32 + 3];
  __shared__ scalar_t kTile[KTILE * KTILE][DIM_32 + 3];

  /* d_out tile */
  const int qtx = lti / DIM_32;
  const int qty = lti - qtx * DIM_32;
  if (qtx < TILE * TILE) {
    int qi = qtx / TILE;
    const int qj = qtx - qi * TILE + sj;
    qi += si;
    if (qi < height && qj < width)
      tile[qtx][qty] = d_out.data()[batchHeadOffset + qi * d_out.stride(2) +
                                    qj * d_out.stride(3) + qty];
  }
  /* value tile */
  const int ktx = lti / KSTRIDE_32;
  const int kty = (lti - ktx * KSTRIDE_32) * KITERS_32;
  if (ktx < KTILE * KTILE) {
    int bi = ktx / KTILE;
    const int bj = ktx - bi * KTILE + snj;
    bi += sni;
    if (bi < height && bj < width) {
      const int valueOffset =
          batchHeadOffset + bi * d_out.stride(2) + bj * d_out.stride(3) + kty;
#pragma unroll
      for (int ti = 0; ti < KITERS_32; ++ti)
        kTile[ktx][kty + ti] = value.data()[valueOffset + ti];
    }
  }
  __syncthreads();
  if (y < height * KERNEL_SIZE && x < width * KERNEL_SIZE) {
    const int i = y / KERNEL_SIZE;
    const int ki = y - i * KERNEL_SIZE;
    const int j = x / KERNEL_SIZE;
    const int kj = x - j * KERNEL_SIZE;
    const int ni = get_window_start(i, height, NEIGHBORHOOD_SIZE);
    const int nj = get_window_start(j, width, NEIGHBORHOOD_SIZE);
    scalar_t updt = scalar_t(0);
    const int d_outIdx = (i - si) * TILE + (j - sj);
    const int valueIdx = (ni + ki - sni) * KTILE + (nj + kj - snj);

#pragma unroll
    for (int dimOffset = 0; dimOffset < DIM_32; ++dimOffset)
      updt += tile[d_outIdx][dimOffset] * kTile[valueIdx][dimOffset];

    const int index = b * d_attn.stride(0) + h * d_attn.stride(1) +
                      i * d_attn.stride(2) + j * d_attn.stride(3) +
                      ki * KERNEL_SIZE + kj;
    d_attn.data()[index] = updt;
  }
  //}
}

template <int TILEX, int TILEY, int KTILEX, int KTILEY, int KERNEL_SIZE,
          int NEIGHBORHOOD_SIZE, int XTHREADS, int YTHREADS, typename scalar_t>
__global__ void nattena_cuda_backward_kernel_fp16_11x11_13x13_32(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        d_out,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        value,
    const int height, const int width, const int batch_size, const int heads) {
  // Because batch heads have stride 1 per threadblock, we can just use blockIdx
  // since blockDim will be 1 and threadIdx will always be 0. const int z =
  // blockIdx.z * blockDim.z + threadIdx.z;
  const int z = blockIdx.z;
  const int b = z / heads;
  const int h = z - b * heads;
  // Not needed again because it will always be true.
  // if (z < batch_size * heads)
  // {
  const int sx = blockIdx.x * XTHREADS;
  const int x = sx + threadIdx.x;
  const int sy = blockIdx.y * YTHREADS;
  const int y = sy + threadIdx.y;
  const int lti = threadIdx.y * XTHREADS + threadIdx.x;
  const int stride2 = DIMHALF_32 * width;
  const int batchHeadOffset =
      b * (stride2 * height * heads) + h * (stride2 * height);
  const int si = sy / KERNEL_SIZE;
  const int sj = sx / KERNEL_SIZE;
  const int sni = get_window_start(si, height, NEIGHBORHOOD_SIZE);
  const int snj = get_window_start(sj, width, NEIGHBORHOOD_SIZE);
  __shared__ __half2 tile[TILEX * TILEY][DIM_32 + 3];
  __shared__ __half2 kTile[KTILEX * KTILEY][DIM_32 + 3];
  __half2 *d_out2 = reinterpret_cast<__half2 *>(d_out.data());
  __half2 *value2 = reinterpret_cast<__half2 *>(value.data());

  /* d_out tile */
  const int qtx = lti / DIM_32;
  const int qtyp = lti - qtx * DIM_32;
  const int qdi = qtyp / KHALFITERS_32;
  const int qdj = qtyp - qdi * KHALFITERS_32;
  const int qty = qdi * KITERS_32 + qdj;
  if (qtx < TILEX * TILEY && qtyp < DIMHALF_32) {
    int qi = qtx / TILEY;
    const int qj = qtx - qi * TILEY + sj;
    qi += si;
    if (qi < height && qj < width)
      tile[qtx][qty] =
          d_out2[batchHeadOffset + qi * stride2 + qj * DIMHALF_32 + qtyp];
  }
  /* value tile */
  const int ktx = lti / KSTRIDE_32;
  const int kty = (lti - ktx * KSTRIDE_32) * KHALFITERS_32;
  if (ktx < KTILEX * KTILEY) {
    int bi = ktx / KTILEY;
    const int bj = ktx - bi * KTILEY + snj;
    bi += sni;
    if (bi < height && bj < width) {
      const int valueOffset =
          batchHeadOffset + bi * stride2 + bj * DIMHALF_32 + kty;
#pragma unroll
      for (int ti = 0; ti < KHALFITERS_32; ++ti)
        kTile[ktx][kty * 2 + ti] = value2[valueOffset + ti];
    }
  }
  __syncthreads();
  if (y < height * KERNEL_SIZE && x < width * KERNEL_SIZE) {
    const int i = y / KERNEL_SIZE;
    const int ki = y - i * KERNEL_SIZE;
    const int j = x / KERNEL_SIZE;
    const int kj = x - j * KERNEL_SIZE;
    const int ni = get_window_start(i, height, NEIGHBORHOOD_SIZE);
    const int nj = get_window_start(j, width, NEIGHBORHOOD_SIZE);
    __half2 updt = __float2half2_rn(0.f);
    const int d_outIdx = (i - si) * TILEY + (j - sj);
    const int valueIdx = (ni + ki - sni) * KTILEY + (nj + kj - snj);

#pragma unroll
    for (int di = 0; di < KSTRIDE_32; ++di)
#pragma unroll
      for (int dj = 0; dj < KHALFITERS_32; ++dj)
        updt = __hfma2(tile[d_outIdx][di * KITERS_32 + dj],
                       kTile[valueIdx][di * KITERS_32 + dj], updt);
    const int index = b * d_attn.stride(0) + h * d_attn.stride(1) +
                      i * d_attn.stride(2) + j * d_attn.stride(3) +
                      ki * KERNEL_SIZE + kj;
    d_attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y));
  }
  //}
}

template <int TILEX, int TILEY, int KTILEX, int KTILEY, int KERNEL_SIZE,
          int NEIGHBORHOOD_SIZE, int XTHREADS, int YTHREADS, typename scalar_t,
          typename memscalar_t>
__global__ void nattena_cuda_backward_kernel_fp32_11x11_13x13_32(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        d_out,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        value,
    const int height, const int width, const int batch_size, const int heads) {
  // Because batch heads have stride 1 per threadblock, we can just use blockIdx
  // since blockDim will be 1 and threadIdx will always be 0. const int z =
  // blockIdx.z * blockDim.z + threadIdx.z;
  const int z = blockIdx.z;
  const int b = z / heads;
  const int h = z - b * heads;
  // Not needed again because it will always be true.
  // if (z < batch_size * heads)
  // {
  const int sx = blockIdx.x * XTHREADS;
  const int x = sx + threadIdx.x;
  const int sy = blockIdx.y * YTHREADS;
  const int y = sy + threadIdx.y;
  const int lti = threadIdx.y * XTHREADS + threadIdx.x;
  const int batchHeadOffset = b * d_out.stride(0) + h * d_out.stride(1);
  const int si = sy / KERNEL_SIZE;
  const int sj = sx / KERNEL_SIZE;
  const int sni = get_window_start(si, height, NEIGHBORHOOD_SIZE);
  const int snj = get_window_start(sj, width, NEIGHBORHOOD_SIZE);
  __shared__ memscalar_t tile[TILEX * TILEY][DIM_32 + 3];
  __shared__ memscalar_t kTile[KTILEX * KTILEY][DIM_32 + 3];

  /* d_out tile */
  const int qtx = lti / DIM_32;
  const int qty = lti - qtx * DIM_32;
  if (qtx < TILEX * TILEY) {
    int qi = qtx / TILEY;
    const int qj = qtx - qi * TILEY + sj;
    qi += si;
    if (qi < height && qj < width)
      tile[qtx][qty] = d_out.data()[batchHeadOffset + qi * d_out.stride(2) +
                                    qj * d_out.stride(3) + qty];
  }
  /* value tile */
  const int ktx = lti / KSTRIDE_32;
  const int kty = (lti - ktx * KSTRIDE_32) * KITERS_32;
  if (ktx < KTILEX * KTILEY) {
    int bi = ktx / KTILEY;
    const int bj = ktx - bi * KTILEY + snj;
    bi += sni;
    if (bi < height && bj < width) {
      const int valueOffset =
          batchHeadOffset + bi * d_out.stride(2) + bj * d_out.stride(3) + kty;
#pragma unroll
      for (int ti = 0; ti < KITERS_32; ++ti)
        kTile[ktx][kty + ti] = value.data()[valueOffset + ti];
    }
  }
  __syncthreads();
  if (y < height * KERNEL_SIZE && x < width * KERNEL_SIZE) {
    const int i = y / KERNEL_SIZE;
    const int ki = y - i * KERNEL_SIZE;
    const int j = x / KERNEL_SIZE;
    const int kj = x - j * KERNEL_SIZE;
    const int ni = get_window_start(i, height, NEIGHBORHOOD_SIZE);
    const int nj = get_window_start(j, width, NEIGHBORHOOD_SIZE);
    scalar_t updt = scalar_t(0);
    const int d_outIdx = (i - si) * TILEY + (j - sj);
    const int valueIdx = (ni + ki - sni) * KTILEY + (nj + kj - snj);

#pragma unroll
    for (int dimOffset = 0; dimOffset < DIM_32; ++dimOffset)
      updt += tile[d_outIdx][dimOffset] * kTile[valueIdx][dimOffset];

    const int index = b * d_attn.stride(0) + h * d_attn.stride(1) +
                      i * d_attn.stride(2) + j * d_attn.stride(3) +
                      ki * KERNEL_SIZE + kj;
    d_attn.data()[index] = updt;
  }
  //}
}

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, typename scalar_t>
__global__ void nattenv_cuda_backward_kernel_fp32(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        d_out,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits> d_value,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        attn,
    const int height, const int width, const int heads, const int dim,
    const int d_value_numel) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearIndex < d_value_numel) {
    int indtmp1 = linearIndex / dim;
    const int d = linearIndex - indtmp1 * dim;
    int indtmp2 = indtmp1 / width;
    const int j = indtmp1 - indtmp2 * width;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1 / height;
    const int i = indtmp1 - indtmp2 * height;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1 / heads;
    const int h = indtmp1 - indtmp2 * heads;
    const int b = indtmp2;
    const int ni = get_backward_window_start(i, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
    const int nj = get_backward_window_start(j, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
    const int ei =
        get_backward_window_end(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
    const int ej =
        get_backward_window_end(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
    const int attnOffset = b * attn.stride(0) + h * attn.stride(1);
    const int outOffset = b * d_out.stride(0) + h * d_out.stride(1) + d;
    scalar_t d_value_update = scalar_t(0);
#pragma unroll
    for (int xi = ni; xi < ei; ++xi) {
      const int oni = get_window_start(xi, height, NEIGHBORHOOD_SIZE);
#pragma unroll
      for (int xj = nj; xj < ej; ++xj) {
        const int onj = get_window_start(xj, width, NEIGHBORHOOD_SIZE);
        const int outIndex =
            outOffset + xi * d_out.stride(2) + xj * d_out.stride(3);
        const int attnIndex = attnOffset + xi * attn.stride(2) +
                              xj * attn.stride(3) + (i - oni) * KERNEL_SIZE +
                              (j - onj);
        d_value_update += d_out.data()[outIndex] * attn.data()[attnIndex];
      }
    }
    d_value.data()[linearIndex] = d_value_update;
  }
}

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, typename scalar_t>
__global__ void nattenv_cuda_backward_kernel_fp16(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        d_out,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits> d_value,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::DefaultPtrTraits>
        attn,
    const int height, const int width, const int heads, const int dimhalf,
    const int d_value_numel) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearIndex < d_value_numel) {
    __half2 *d_out2 = reinterpret_cast<__half2 *>(d_out.data());
    __half2 *d_value2 = reinterpret_cast<__half2 *>(d_value.data());
    int indtmp1 = linearIndex / dimhalf;
    const int d = linearIndex - indtmp1 * dimhalf;
    int indtmp2 = indtmp1 / width;
    const int j = indtmp1 - indtmp2 * width;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1 / height;
    const int i = indtmp1 - indtmp2 * height;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1 / heads;
    const int h = indtmp1 - indtmp2 * heads;
    const int b = indtmp2;
    const int ni = get_backward_window_start(i, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
    const int nj = get_backward_window_start(j, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
    const int ei =
        get_backward_window_end(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
    const int ej =
        get_backward_window_end(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE);
    const int attnOffset = b * attn.stride(0) + h * attn.stride(1);
    const int stride2 = dimhalf * width;
    const int outOffset =
        b * (stride2 * height * heads) + h * (stride2 * height) + d;
    __half2 d_value_update = __float2half2_rn(0.f);
#pragma unroll
    for (int xi = ni; xi < ei; ++xi) {
      const int oni = get_window_start(xi, height, NEIGHBORHOOD_SIZE);
#pragma unroll
      for (int xj = nj; xj < ej; ++xj) {
        const int onj = get_window_start(xj, width, NEIGHBORHOOD_SIZE);
        const int outIndex = outOffset + xi * stride2 + xj * dimhalf;
        const int attnIndex = attnOffset + xi * attn.stride(2) +
                              xj * attn.stride(3) + (i - oni) * KERNEL_SIZE +
                              (j - onj);
        scalar_t a = attn.data()[attnIndex];
        d_value_update =
            __hfma2(d_out2[outIndex], __halves2half2(a, a), d_value_update);
      }
    }
    d_value2[linearIndex] = d_value_update;
  }
}

torch::Tensor nattenav_cuda_forward_fp16(const torch::Tensor &attn,
                                         const torch::Tensor &value) {
  at::cuda::CUDAGuard device_guard(value.device());
  int batch_size = value.size(0);
  int heads = value.size(1);
  int height = value.size(2);
  int width = value.size(3);
  int dimhalf = value.size(4) / 2;
  TORCH_CHECK(dimhalf * 2 == value.size(4),
              "Dims per head must be an even number in FP16.");
  int kernel_size_sq = attn.size(4);
  int kernel_size = sqrt(kernel_size_sq);
  CHECK_FEATMAP(height, width, kernel_size);
  CHECK_KERNELSIZE("nattenav_cuda_forward_fp16", kernel_size);

  auto out = torch::zeros_like(value);

  int32_t nhalf = out.numel() / 2;
  int blocks = GET_BLOCKS(nhalf, CUDA_NUM_THREADS_FP16, -1);
  dim3 grid(blocks);
  dim3 block(CUDA_NUM_THREADS_FP16);
  const auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_HALF_TYPES(
      at::kHalf, value.scalar_type(), "nattenav_forward_cuda_fp16", ([&] {
        const auto attn_a =
            attn.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        const auto value_a =
            value.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        auto out_a =
            out.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS(kernel_size, nattenav_cuda_forward_kernel_fp16, grid,
                       block, 0, stream, attn_a, value_a, out_a, height, width,
                       heads, dimhalf, nhalf);
      }));
  return out;
}

torch::Tensor nattenav_cuda_forward(const torch::Tensor &attn,
                                    const torch::Tensor &value) {
  at::cuda::CUDAGuard device_guard(value.device());
  int batch_size = value.size(0);
  int heads = value.size(1);
  int height = value.size(2);
  int width = value.size(3);
  int dim = value.size(4);
  int kernel_size_sq = attn.size(4);
  int kernel_size = sqrt(kernel_size_sq);
  CHECK_FEATMAP(height, width, kernel_size);
  CHECK_KERNELSIZE("nattenav_cuda_forward", kernel_size);

  auto out = torch::zeros_like(value);

  int32_t n = out.numel();
  int blocks = GET_BLOCKS(n, CUDA_NUM_THREADS_F, -1);
  dim3 grid(blocks);
  dim3 block(CUDA_NUM_THREADS_F);
  const auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      value.scalar_type(), "nattenav_forward_cuda", ([&] {
        const auto attn_a =
            attn.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        const auto value_a =
            value.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        auto out_a =
            out.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS(kernel_size, nattenav_cuda_forward_kernel_fp32, grid,
                       block, 0, stream, attn_a, value_a, out_a, height, width,
                       heads, dim, n);
      }));
  return out;
}

std::vector<torch::Tensor> nattenav_cuda_backward_tiled_32(
    const torch::Tensor &d_out, const torch::Tensor &attn,
    const torch::Tensor &value) {
  at::cuda::CUDAGuard device_guard(value.device());
  int64_t batch_size = value.size(0);
  int64_t heads = value.size(1);
  int64_t height = value.size(2);
  int64_t width = value.size(3);
  int64_t dim = value.size(4);
  int64_t kernel_size_sq = attn.size(4);
  int kernel_size = sqrt(kernel_size_sq);
  int xsize = width * kernel_size;
  int ysize = height * kernel_size;
  int zsize = batch_size * heads;
  CHECK_FEATMAP(height, width, kernel_size);
  TORCH_CHECK(dim == DIM_32, "nattenav_cuda_backward_tiled_32",
              " only supports 32-dim attention heads.");
  TORCH_CHECK(kernel_size == KERNEL_SIZE_7 || kernel_size == KERNEL_SIZE_5 ||
                  kernel_size == KERNEL_SIZE_9 ||
                  kernel_size == KERNEL_SIZE_11 ||
                  kernel_size == KERNEL_SIZE_13,
              "nattenav_cuda_backward_tiled_32",
              " only supports kernel sizes 5, 7, 9, 11, and 13.");

  auto d_attn = torch::zeros_like(attn);
  auto d_value = torch::zeros_like(value);
  int XTHREADS = -1;
  int YTHREADS = -1;
  int BATCHTHREADS = -1;
  if (kernel_size == KERNEL_SIZE_7) {
    XTHREADS = XYTHREADS_7;
    YTHREADS = XYTHREADS_7;
    BATCHTHREADS = BATCHTHREADS_7;
  } else if (kernel_size == KERNEL_SIZE_5) {
    XTHREADS = XYTHREADS_5;
    YTHREADS = XYTHREADS_5;
    BATCHTHREADS = BATCHTHREADS_5;
  } else if (kernel_size == KERNEL_SIZE_9) {
    XTHREADS = XYTHREADS_9;
    YTHREADS = XYTHREADS_9;
    BATCHTHREADS = BATCHTHREADS_9;
  } else if (kernel_size == KERNEL_SIZE_11) {
    XTHREADS = XTHREADS_11;
    YTHREADS = YTHREADS_11;
    BATCHTHREADS = BATCHTHREADS_11;
  } else if (kernel_size == KERNEL_SIZE_13) {
    XTHREADS = XTHREADS_13;
    YTHREADS = YTHREADS_13;
    BATCHTHREADS = BATCHTHREADS_13;
  }
  const dim3 attn_blocks((xsize + XTHREADS - 1) / XTHREADS,
                         (ysize + YTHREADS - 1) / YTHREADS,
                         (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
  const dim3 attn_threads(XTHREADS, YTHREADS, BATCHTHREADS);

  int32_t n_value = d_value.numel();
  int blocks_value = GET_BLOCKS(n_value, CUDA_NUM_THREADS_V, -1);
  dim3 grid_value(blocks_value);
  dim3 block(CUDA_NUM_THREADS_V);
  const auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      d_attn.scalar_type(), "nattenav_cuda_backward_tiled_32", ([&] {
        auto d_attn_a =
            d_attn.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        auto d_value_a =
            d_value.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        const auto d_out_a =
            d_out.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        const auto value_a =
            value.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        const auto attn_a =
            attn.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        if (kernel_size == KERNEL_SIZE_7) {
          nattena_cuda_backward_kernel_fp32_7x7_9x9_32<
              TILE_7, KTILE_7, KERNEL_SIZE_7, NEIGHBORHOOD_SIZE_7, XYTHREADS_7,
              scalar_t><<<attn_blocks, attn_threads, 0, stream>>>(
              d_out_a, d_attn_a, value_a, height, width, batch_size, heads);
          nattenv_cuda_backward_kernel_fp32<KERNEL_SIZE_7, NEIGHBORHOOD_SIZE_7,
                                            scalar_t>
              <<<grid_value, block, 0, stream>>>(d_out_a, d_value_a, attn_a,
                                                 height, width, heads, dim,
                                                 n_value);
        } else if (kernel_size == KERNEL_SIZE_9) {
          nattena_cuda_backward_kernel_fp32_7x7_9x9_32<
              TILE_9, KTILE_9, KERNEL_SIZE_9, NEIGHBORHOOD_SIZE_9, XYTHREADS_9,
              scalar_t><<<attn_blocks, attn_threads, 0, stream>>>(
              d_out_a, d_attn_a, value_a, height, width, batch_size, heads);
          nattenv_cuda_backward_kernel_fp32<KERNEL_SIZE_9, NEIGHBORHOOD_SIZE_9,
                                            scalar_t>
              <<<grid_value, block, 0, stream>>>(d_out_a, d_value_a, attn_a,
                                                 height, width, heads, dim,
                                                 n_value);
        } else if (kernel_size == KERNEL_SIZE_5) {
          nattena_cuda_backward_kernel_fp32_5x5_32<scalar_t>
              <<<attn_blocks, attn_threads, 0, stream>>>(
                  d_out_a, d_attn_a, value_a, height, width, batch_size, heads);
          nattenv_cuda_backward_kernel_fp32<KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5,
                                            scalar_t>
              <<<grid_value, block, 0, stream>>>(d_out_a, d_value_a, attn_a,
                                                 height, width, heads, dim,
                                                 n_value);
        } else if (kernel_size == KERNEL_SIZE_11) {
          nattena_cuda_backward_kernel_fp32_11x11_13x13_32<
              TILE_11_X, TILE_11_Y, KTILE_11_X, KTILE_11_Y, KERNEL_SIZE_11,
              NEIGHBORHOOD_SIZE_11, XTHREADS_11, YTHREADS_11, scalar_t,
              scalar_t><<<attn_blocks, attn_threads, 0, stream>>>(
              d_out_a, d_attn_a, value_a, height, width, batch_size, heads);
          nattenv_cuda_backward_kernel_fp32<KERNEL_SIZE_11,
                                            NEIGHBORHOOD_SIZE_11, scalar_t>
              <<<grid_value, block, 0, stream>>>(d_out_a, d_value_a, attn_a,
                                                 height, width, heads, dim,
                                                 n_value);
        } else if (kernel_size == KERNEL_SIZE_13) {
          nattena_cuda_backward_kernel_fp32_11x11_13x13_32<
              TILE_13_X, TILE_13_Y, KTILE_13_X, KTILE_13_Y, KERNEL_SIZE_13,
              NEIGHBORHOOD_SIZE_13, XTHREADS_13, YTHREADS_13, scalar_t, float>
              <<<attn_blocks, attn_threads, 0, stream>>>(
                  d_out_a, d_attn_a, value_a, height, width, batch_size, heads);
          nattenv_cuda_backward_kernel_fp32<KERNEL_SIZE_13,
                                            NEIGHBORHOOD_SIZE_13, scalar_t>
              <<<grid_value, block, 0, stream>>>(d_out_a, d_value_a, attn_a,
                                                 height, width, heads, dim,
                                                 n_value);
        }
      }));
  return {d_attn, d_value};
}

std::vector<torch::Tensor> nattenav_cuda_backward_fp16_tiled_32(
    const torch::Tensor &d_out, const torch::Tensor &attn,
    const torch::Tensor &value) {
  at::cuda::CUDAGuard device_guard(value.device());
  int64_t batch_size = value.size(0);
  int64_t heads = value.size(1);
  int64_t height = value.size(2);
  int64_t width = value.size(3);
  int64_t dimhalf = value.size(4) / 2;
  TORCH_CHECK(dimhalf * 2 == value.size(4),
              "Dims per head must be an even number in FP16.");
  int64_t kernel_size_sq = attn.size(4);
  int kernel_size = sqrt(kernel_size_sq);
  int xsize = width * kernel_size;
  int ysize = height * kernel_size;
  int zsize = batch_size * heads;
  CHECK_FEATMAP(height, width, kernel_size);
  TORCH_CHECK(dimhalf * 2 == DIM_32, "nattenav_cuda_backward_fp16_tiled_32",
              " only supports 32-dim attention heads.");
  TORCH_CHECK(kernel_size == KERNEL_SIZE_7 || kernel_size == KERNEL_SIZE_5 ||
                  kernel_size == KERNEL_SIZE_9 ||
                  kernel_size == KERNEL_SIZE_11 ||
                  kernel_size == KERNEL_SIZE_13,
              "nattenav_cuda_backward_fp16_tiled_32",
              " only supports kernel sizes 5, 7, 9, 11, and 13.");

  auto d_attn = torch::zeros_like(attn);
  auto d_value = torch::zeros_like(value);
  int XTHREADS = -1;
  int YTHREADS = -1;
  int BATCHTHREADS = -1;
  if (kernel_size == KERNEL_SIZE_7) {
    XTHREADS = XYTHREADS_7;
    YTHREADS = XYTHREADS_7;
    BATCHTHREADS = BATCHTHREADS_7;
  } else if (kernel_size == KERNEL_SIZE_5) {
    XTHREADS = XYTHREADS_5;
    YTHREADS = XYTHREADS_5;
    BATCHTHREADS = BATCHTHREADS_5;
  } else if (kernel_size == KERNEL_SIZE_9) {
    XTHREADS = XYTHREADS_9;
    YTHREADS = XYTHREADS_9;
    BATCHTHREADS = BATCHTHREADS_9;
  } else if (kernel_size == KERNEL_SIZE_11) {
    XTHREADS = XTHREADS_11;
    YTHREADS = YTHREADS_11;
    BATCHTHREADS = BATCHTHREADS_11;
  } else if (kernel_size == KERNEL_SIZE_13) {
    XTHREADS = XTHREADS_13;
    YTHREADS = YTHREADS_13;
    BATCHTHREADS = BATCHTHREADS_13;
  }
  const dim3 attn_blocks((xsize + XTHREADS - 1) / XTHREADS,
                         (ysize + YTHREADS - 1) / YTHREADS,
                         (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
  const dim3 attn_threads(XTHREADS, YTHREADS, BATCHTHREADS);

  int32_t nhalf_value = d_value.numel() / 2;
  int blocks_value = GET_BLOCKS(nhalf_value, CUDA_NUM_THREADS_V16, -1);
  dim3 grid_value(blocks_value);
  dim3 block(CUDA_NUM_THREADS_V16);
  const auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_HALF_TYPES(
      at::kHalf, d_attn.scalar_type(), "nattenav_cuda_backward_fp16_tiled_32",
      ([&] {
        auto d_attn_a =
            d_attn.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        auto d_value_a =
            d_value.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        const auto d_out_a =
            d_out.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        const auto value_a =
            value.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        const auto attn_a =
            attn.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        if (kernel_size == KERNEL_SIZE_7) {
          nattena_cuda_backward_kernel_fp16_7x7_9x9_32<
              TILE_7, KTILE_7, KERNEL_SIZE_7, NEIGHBORHOOD_SIZE_7, XYTHREADS_7,
              scalar_t><<<attn_blocks, attn_threads, 0, stream>>>(
              d_out_a, d_attn_a, value_a, height, width, batch_size, heads);
          nattenv_cuda_backward_kernel_fp16<KERNEL_SIZE_7, NEIGHBORHOOD_SIZE_7,
                                            scalar_t>
              <<<grid_value, block, 0, stream>>>(d_out_a, d_value_a, attn_a,
                                                 height, width, heads, dimhalf,
                                                 nhalf_value);
        } else if (kernel_size == KERNEL_SIZE_9) {
          nattena_cuda_backward_kernel_fp16_7x7_9x9_32<
              TILE_9, KTILE_9, KERNEL_SIZE_9, NEIGHBORHOOD_SIZE_9, XYTHREADS_9,
              scalar_t><<<attn_blocks, attn_threads, 0, stream>>>(
              d_out_a, d_attn_a, value_a, height, width, batch_size, heads);
          nattenv_cuda_backward_kernel_fp16<KERNEL_SIZE_9, NEIGHBORHOOD_SIZE_9,
                                            scalar_t>
              <<<grid_value, block, 0, stream>>>(d_out_a, d_value_a, attn_a,
                                                 height, width, heads, dimhalf,
                                                 nhalf_value);
        } else if (kernel_size == KERNEL_SIZE_5) {
          nattena_cuda_backward_kernel_fp16_5x5_32<scalar_t>
              <<<attn_blocks, attn_threads, 0, stream>>>(
                  d_out_a, d_attn_a, value_a, height, width, batch_size, heads);
          nattenv_cuda_backward_kernel_fp16<KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5,
                                            scalar_t>
              <<<grid_value, block, 0, stream>>>(d_out_a, d_value_a, attn_a,
                                                 height, width, heads, dimhalf,
                                                 nhalf_value);
        } else if (kernel_size == KERNEL_SIZE_11) {
          nattena_cuda_backward_kernel_fp16_11x11_13x13_32<
              TILE_11_X, TILE_11_Y, KTILE_11_X, KTILE_11_Y, KERNEL_SIZE_11,
              NEIGHBORHOOD_SIZE_11, XTHREADS_11, YTHREADS_11, scalar_t>
              <<<attn_blocks, attn_threads, 0, stream>>>(
                  d_out_a, d_attn_a, value_a, height, width, batch_size, heads);
          nattenv_cuda_backward_kernel_fp16<KERNEL_SIZE_11,
                                            NEIGHBORHOOD_SIZE_11, scalar_t>
              <<<grid_value, block, 0, stream>>>(d_out_a, d_value_a, attn_a,
                                                 height, width, heads, dimhalf,
                                                 nhalf_value);
        } else if (kernel_size == KERNEL_SIZE_13) {
          nattena_cuda_backward_kernel_fp16_11x11_13x13_32<
              TILE_13_X, TILE_13_Y, KTILE_13_X, KTILE_13_Y, KERNEL_SIZE_13,
              NEIGHBORHOOD_SIZE_13, XTHREADS_13, YTHREADS_13, scalar_t>
              <<<attn_blocks, attn_threads, 0, stream>>>(
                  d_out_a, d_attn_a, value_a, height, width, batch_size, heads);
          nattenv_cuda_backward_kernel_fp16<KERNEL_SIZE_13,
                                            NEIGHBORHOOD_SIZE_13, scalar_t>
              <<<grid_value, block, 0, stream>>>(d_out_a, d_value_a, attn_a,
                                                 height, width, heads, dimhalf,
                                                 nhalf_value);
        }
      }));
  return {d_attn, d_value};
}

std::vector<torch::Tensor> nattenav_cuda_backward(const torch::Tensor &d_out,
                                                  const torch::Tensor &attn,
                                                  const torch::Tensor &value) {
  at::cuda::CUDAGuard device_guard(value.device());
  int64_t batch_size = value.size(0);
  int64_t heads = value.size(1);
  int64_t height = value.size(2);
  int64_t width = value.size(3);
  int64_t dim = value.size(4);
  int kernel_size_sq = attn.size(4);
  int kernel_size = sqrt(kernel_size_sq);
  int zsize = batch_size * heads;
  int xsize = height * width;
  CHECK_FEATMAP(height, width, kernel_size);
  CHECK_KERNELSIZE("nattenav_cuda_backward", kernel_size);
  int KERNELTHREADS = min(CUDA_NUM_THREADS, kernel_size_sq);
  int PIXELTHREADS = min(int(CUDA_NUM_THREADS / KERNELTHREADS), xsize);
  int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (PIXELTHREADS * KERNELTHREADS));

  auto d_attn = torch::zeros_like(attn);
  auto d_value = torch::zeros_like(value);

  const dim3 attn_blocks((xsize + PIXELTHREADS - 1) / PIXELTHREADS,
                         (kernel_size_sq + KERNELTHREADS - 1) / KERNELTHREADS,
                         (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
  const dim3 attn_threads(PIXELTHREADS, KERNELTHREADS, BATCHTHREADS);
  int32_t n_value = d_value.numel();
  int blocks_value = GET_BLOCKS(n_value, CUDA_NUM_THREADS_F, -1);
  dim3 grid_value(blocks_value);
  dim3 block(CUDA_NUM_THREADS);
  const auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      d_attn.scalar_type(), "nattenav_backward_cuda", ([&] {
        auto d_attn_a =
            d_attn.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        auto d_value_a =
            d_value.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        const auto d_out_a =
            d_out.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        const auto value_a =
            value.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        const auto attn_a =
            attn.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS(kernel_size, nattena_cuda_backward_kernel_fp32,
                       attn_blocks, attn_threads, 0, stream, d_out_a, d_attn_a,
                       value_a, height, width, batch_size, heads, dim);
        LAUNCH_DNA_KNS(kernel_size, nattenv_cuda_backward_kernel_fp32,
                       grid_value, block, 0, stream, d_out_a, d_value_a, attn_a,
                       height, width, heads, dim, n_value);
      }));
  return {d_attn, d_value};
}

std::vector<torch::Tensor> nattenav_cuda_backward_fp16(
    const torch::Tensor &d_out, const torch::Tensor &attn,
    const torch::Tensor &value) {
  at::cuda::CUDAGuard device_guard(value.device());
  int64_t batch_size = value.size(0);
  int64_t heads = value.size(1);
  int64_t height = value.size(2);
  int64_t width = value.size(3);
  int64_t dimhalf = value.size(4) / 2;
  TORCH_CHECK(dimhalf * 2 == value.size(4),
              "Dims per head must be an even number in FP16.");
  int kernel_size_sq = attn.size(4);
  int kernel_size = sqrt(kernel_size_sq);
  int zsize = batch_size * heads;
  int xsize = height * width;
  CHECK_FEATMAP(height, width, kernel_size);
  CHECK_KERNELSIZE("nattenav_cuda_backward_fp16", kernel_size);
  int KERNELTHREADS = min(CUDA_NUM_THREADS, kernel_size_sq);
  int PIXELTHREADS = min(int(CUDA_NUM_THREADS / KERNELTHREADS), xsize);
  int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (PIXELTHREADS * KERNELTHREADS));

  auto d_attn = torch::zeros_like(attn);
  auto d_value = torch::zeros_like(value);

  const dim3 attn_blocks((xsize + PIXELTHREADS - 1) / PIXELTHREADS,
                         (kernel_size_sq + KERNELTHREADS - 1) / KERNELTHREADS,
                         (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
  const dim3 attn_threads(PIXELTHREADS, KERNELTHREADS, BATCHTHREADS);
  int32_t nhalf_value = d_value.numel() / 2;
  int blocks_value = GET_BLOCKS(nhalf_value, CUDA_NUM_THREADS_FP16, -1);
  dim3 grid_value(blocks_value);
  dim3 block(CUDA_NUM_THREADS);
  const auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_HALF_TYPES(
      at::kHalf, d_attn.scalar_type(), "nattenav_backward_cuda_fp16", ([&] {
        auto d_attn_a =
            d_attn.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        auto d_value_a =
            d_value.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        const auto d_out_a =
            d_out.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        const auto value_a =
            value.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        const auto attn_a =
            attn.packed_accessor32<scalar_t, 5, torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS(kernel_size, nattena_cuda_backward_kernel_fp16,
                       attn_blocks, attn_threads, 0, stream, d_out_a, d_attn_a,
                       value_a, height, width, batch_size, heads, dimhalf);
        LAUNCH_DNA_KNS(kernel_size, nattenv_cuda_backward_kernel_fp16,
                       grid_value, block, 0, stream, d_out_a, d_value_a, attn_a,
                       height, width, heads, dimhalf, nhalf_value);
      }));
  return {d_attn, d_value};
}

// C++ interface
torch::Tensor NATTENAVForwardCUDAKernelLauncher(const torch::Tensor attn,
                                                const torch::Tensor value) {
  CHECK_CUDA_INPUT(attn);
  CHECK_CUDA_INPUT(value);
  bool half =
      ::detail::scalar_type(value.scalar_type()) == at::ScalarType::Half;
  if (half) return nattenav_cuda_forward_fp16(attn, value);
  return nattenav_cuda_forward(attn, value);
}

std::vector<torch::Tensor> NATTENAVBackwardCUDAKernelLauncher(
    const torch::Tensor d_out, const torch::Tensor attn,
    const torch::Tensor value) {
  CHECK_CUDA_INPUT(d_out);
  CHECK_CUDA_INPUT(attn);
  CHECK_CUDA_INPUT(value);
  int dim = value.size(4);
  int kernel_size = sqrt(attn.size(4));
  bool half =
      ::detail::scalar_type(value.scalar_type()) == at::ScalarType::Half;
  if ((kernel_size == 7 || kernel_size == 5 || kernel_size == 9 ||
       kernel_size == 11 || kernel_size == 13) &&
      dim == 32) {
    if (half) return nattenav_cuda_backward_fp16_tiled_32(d_out, attn, value);
    return nattenav_cuda_backward_tiled_32(d_out, attn, value);
  }
  if (half) return nattenav_cuda_backward_fp16(d_out, attn, value);
  return nattenav_cuda_backward(d_out, attn, value);
}
