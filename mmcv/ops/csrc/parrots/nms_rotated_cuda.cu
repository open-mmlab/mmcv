// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/nms_rotated/nms_rotated_cuda.cu
#include "nms_rotated_cuda.cuh"
#include "parrots_cuda_helper.hpp"

DArrayLite nms_rotated_cuda(const DArrayLite dets, const DArrayLite scores,
                            const DArrayLite dets_sorted, float iou_threshold,
                            const int multi_label, cudaStream_t stream,
                            CudaContext& ctx) {
  int dets_num = dets.dim(0);

  const int col_blocks = divideUP(dets_num, threadsPerBlock);

  auto mask = ctx.createDArrayLite(
      DArraySpec::array(Prim::Int64, DArrayShape(dets_num * col_blocks)));

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(dets_sorted.elemType().prim(), [&] {
    nms_rotated_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        dets_num, iou_threshold, dets_sorted.ptr<scalar_t>(),
        (unsigned long long*)mask.ptr<int64_t>(), multi_label);
  });

  DArrayLite mask_cpu = ctx.createDArrayLite(mask, getHostProxy());
  unsigned long long* mask_host = (unsigned long long*)mask_cpu.ptr<int64_t>();

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  auto keep = ctx.createDArrayLite(
      DArraySpec::array(Prim::Int64, DArrayShape(dets_num)), getHostProxy());

  int64_t* keep_out = keep.ptr<int64_t>();

  for (int i = 0; i < dets_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[i] = 1;
      unsigned long long* p = mask_host + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  auto keep_cuda = ctx.createDArrayLite(keep, ctx.getProxy());
  PARROTS_CUDA_CHECK(cudaGetLastError());
  return keep_cuda;
}
