// Modified from
// https://github.com/Turoad/CLRNet by VBTI

// The functions below originates from Fast R-CNN
// See https://github.com/rbgirshick/py-faster-rcnn
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License
// Written by Shaoqing Ren

#include <torch/extension.h>

#include "line_nms_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

#define STRIDE 4

#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

std::vector<Tensor> LineNMSForwardCudaKernelLauncher(Tensor boxes, Tensor idx,
                                                     float nms_overlap_thresh,
                                                     unsigned long top_k) {
  // initialize
  at::cuda::CUDAGuard device_guard(boxes.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const auto boxes_num = boxes.size(0);
  TORCH_CHECK(boxes.size(1) == PROP_SIZE,
              "Wrong number of offsets. Please adjust `PROP_SIZE`");

  const int col_blocks = DIVUP2(boxes_num, threadsPerBlock);

  AT_ASSERTM(col_blocks < MAX_COL_BLOCKS,
             "The number of column blocks must be less than MAX_COL_BLOCKS. "
             "Increase the MAX_COL_BLOCKS constant if needed.");

  auto longOptions =
      torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong);
  auto mask = at::empty({boxes_num * col_blocks}, longOptions);

  const int col_blocks_alloc = GET_BLOCKS(boxes_num, threadsPerBlock);
  dim3 blocks(col_blocks_alloc, col_blocks_alloc);
  dim3 threads(threadsPerBlock);

  CHECK_CONTIGUOUS(boxes);
  CHECK_CONTIGUOUS(idx);
  CHECK_CONTIGUOUS(mask);

  // Launch the NMS kernel
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      boxes.scalar_type(), "line_nms_cuda_forward_cuda_kernel", ([&] {
        line_nms_cuda_forward_cuda_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                boxes_num, (scalar_t)nms_overlap_thresh,
                boxes.data_ptr<scalar_t>(), idx.data_ptr<int64_t>(),
                mask.data_ptr<int64_t>());
      }));

  // Create output tensors
  auto keep = at::empty({boxes_num}, longOptions);
  auto parent_object_index = at::empty({boxes_num}, longOptions);
  auto num_to_keep = at::empty({}, longOptions);

  // Launch the collect kernel
  nms_collect<<<1, min(col_blocks, THREADS_PER_BLOCK),
                col_blocks * sizeof(unsigned long long), stream>>>(
      boxes_num, col_blocks, top_k, idx.data_ptr<int64_t>(),
      mask.data_ptr<int64_t>(), keep.data_ptr<int64_t>(),
      parent_object_index.data_ptr<int64_t>(), num_to_keep.data_ptr<int64_t>());

  // // Check for CUDA errors
  AT_CUDA_CHECK(cudaGetLastError());

  return {keep, num_to_keep, parent_object_index};
}
