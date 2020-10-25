// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include "pytorch_cuda_helper.hpp"
#include "nms_rotated_cuda.cuh"

Tensor nms_rotated_cuda(
    const Tensor dets,
    const Tensor scores,
    float iou_threshold) {
  // using scalar_t = float;
  AT_ASSERTM(dets.type().is_cuda(), "dets must be a CUDA tensor");
  AT_ASSERTM(scores.type().is_cuda(), "scores must be a CUDA tensor");
  at::cuda::CUDAGuard device_guard(dets.device());

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
  auto dets_sorted = dets.index_select(0, order_t);

  int dets_num = dets.size(0);

  const int col_blocks = at::cuda::ATenCeilDiv(dets_num, threadsPerBlock);

  Tensor mask =
      at::empty({dets_num * col_blocks}, dets.options().dtype(at::kLong));

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      dets_sorted.type(), "nms_rotated_kernel_cuda", [&] {
        nms_rotated_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            dets_num,
            iou_threshold,
            dets_sorted.data<scalar_t>(),
            (unsigned long long*)mask.data<int64_t>());
      });

  Tensor mask_cpu = mask.to(at::kCPU);
  unsigned long long* mask_host = (unsigned long long*)mask_cpu.data<int64_t>();

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  Tensor keep =
      at::empty({dets_num}, dets.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = keep.data<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < dets_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long* p = mask_host + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return order_t.index(
      {keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep)
           .to(order_t.device(), keep.scalar_type())});
}
