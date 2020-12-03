// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_cuda.cu
#include "box_iou_rotated_cuda.cuh"
#include "parrots_cuda_helper.hpp"

DArrayLite box_iou_rotated_cuda(const DArrayLite boxes1,
                                const DArrayLite boxes2, cudaStream_t stream,
                                CudaContext& ctx) {
  using scalar_t = float;

  int num_boxes1 = boxes1.dim(0);
  int num_boxes2 = boxes2.dim(0);

  auto ious = ctx.createDArrayLite(
      DArraySpec::array(Prim::Float32, DArrayShape(num_boxes1 * num_boxes2)));

  if (num_boxes1 > 0 && num_boxes2 > 0) {
    const int blocks_x = divideUP(num_boxes1, BLOCK_DIM_X);
    const int blocks_y = divideUP(num_boxes2, BLOCK_DIM_Y);

    dim3 blocks(blocks_x, blocks_y);
    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);

    box_iou_rotated_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        num_boxes1, num_boxes2, boxes1.ptr<scalar_t>(), boxes2.ptr<scalar_t>(),
        (scalar_t*)ious.ptr<scalar_t>());

    PARROTS_CUDA_CHECK(cudaGetLastError());
  }

  return ious.view({num_boxes1, num_boxes2});
}
