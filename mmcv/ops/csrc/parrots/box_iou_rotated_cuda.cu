// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_cuda.cu
#include "box_iou_rotated_cuda.cuh"
#include "parrots_cuda_helper.hpp"

void box_iou_rotated_cuda_launcher(const DArrayLite boxes1,
                                   const DArrayLite boxes2, DArrayLite ious,
                                   const int mode_flag, const bool aligned,
                                   cudaStream_t stream) {
  using scalar_t = float;

  int output_size = ious.size();
  int num_boxes1 = boxes1.dim(0);
  int num_boxes2 = boxes2.dim(0);

  box_iou_rotated_cuda_kernel<scalar_t>
      <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
          num_boxes1, num_boxes2, boxes1.ptr<scalar_t>(),
          boxes2.ptr<scalar_t>(), (scalar_t*)ious.ptr<scalar_t>(), mode_flag,
          aligned);

  PARROTS_CUDA_CHECK(cudaGetLastError());
}
