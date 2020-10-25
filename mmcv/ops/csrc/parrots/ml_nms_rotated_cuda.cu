// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include "parrots_cuda_helper.hpp"
#include "ml_nms_rotated_cuda.cuh"
#include <iostream>

DArrayLite nms_rotated_cuda(
    const DArrayLite dets,
    const DArrayLite scores,
    const DArrayLite labels,
    const DArrayLite dets_sorted,
    const float iou_threshold,
    cudaStream_t stream,
    CudaContext& ctx) {

  int dets_num = dets.dim(0);

  const int col_blocks = divideUP(dets_num, threadsPerBlock);

  auto mask = ctx.createDArrayLite(
    DArraySpec::array(Prim::Int64, DArrayShape(dets_num * col_blocks)));

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      dets_sorted.elemType().prim(), [&] {
        nms_rotated_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            dets_num,
            iou_threshold,
            dets_sorted.ptr<scalar_t>(),
            (unsigned long long*)mask.ptr<int64_t>());
      });
  
  DArrayLite mask_cpu = ctx.createDArrayLite(mask, getHostProxy());
  unsigned long long* mask_host = (unsigned long long*)mask_cpu.ptr<int64_t>();

  auto remv = ctx.createDArrayLite(DArraySpec::array(Prim::Int64, col_blocks),
                                   getHostProxy());
  remv.setZeros(syncStream());
  auto remv_ptr = remv.ptr<int64_t>();

  auto keep = ctx.createDArrayLite(
    DArraySpec::array(Prim::Int64, DArrayShape(dets_num)), getHostProxy());

  auto keep_out = keep.ptr<int64_t>();

  for (int i = 0; i < dets_num; i++) {

    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;
    if (!(remv_ptr[nblock] & (1ULL << inblock))) {
      keep_out[i] = 1;
      unsigned long long* p = mask_host + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv_ptr[j] |= p[j];
      }
    }
  }

  auto keep_cuda = ctx.createDArrayLite(keep, ctx.getProxy());
  PARROTS_CUDA_CHECK(cudaGetLastError());
  return keep_cuda;
}
