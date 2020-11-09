// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_cuda.cu
#include "box_iou_rotated_cuda.cuh"
#include "pytorch_cuda_helper.hpp"

Tensor box_iou_rotated_cuda(const Tensor boxes1, const Tensor boxes2) {
  using scalar_t = float;
  AT_ASSERTM(boxes1.type().is_cuda(), "boxes1 must be a CUDA tensor");
  AT_ASSERTM(boxes2.type().is_cuda(), "boxes2 must be a CUDA tensor");
  at::cuda::CUDAGuard device_guard(boxes1.device());

  int num_boxes1 = boxes1.size(0);
  int num_boxes2 = boxes2.size(0);

  Tensor ious =
      at::empty({num_boxes1 * num_boxes2}, boxes1.options().dtype(at::kFloat));

  if (num_boxes1 > 0 && num_boxes2 > 0) {
    const int blocks_x = at::cuda::ATenCeilDiv(num_boxes1, BLOCK_DIM_X);
    const int blocks_y = at::cuda::ATenCeilDiv(num_boxes2, BLOCK_DIM_Y);

    dim3 blocks(blocks_x, blocks_y);
    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    box_iou_rotated_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        num_boxes1, num_boxes2, boxes1.data_ptr<scalar_t>(),
        boxes2.data_ptr<scalar_t>(), (scalar_t*)ious.data_ptr<scalar_t>());

    AT_CUDA_CHECK(cudaGetLastError());
  }

  // reshape from 1d array to 2d array
  auto shape = std::vector<int64_t>{num_boxes1, num_boxes2};
  return ious.reshape(shape);
}
