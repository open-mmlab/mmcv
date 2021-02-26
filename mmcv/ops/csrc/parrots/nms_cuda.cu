#include "nms_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

Tensor NMSCUDAKernelLauncher(Tensor boxes, Tensor scores, float iou_threshold,
                             int offset) {
  at::cuda::CUDAGuard device_guard(boxes.device());

  if (boxes.numel() == 0) {
    return at::empty({0}, boxes.options().dtype(at::kLong));
  }
  auto order_t = std::get<1>(scores.sort(0, /*descending=*/true));
  auto boxes_sorted = boxes.index_select(0, order_t);

  int boxes_num = boxes.size(0);
  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
  Tensor mask =
      at::empty({boxes_num, col_blocks}, boxes.options().dtype(at::kLong));
  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  nms_cuda<<<blocks, threads, 0, stream>>>(
      boxes_num, iou_threshold, offset, boxes_sorted.data_ptr<float>(),
      (unsigned long long*)mask.data_ptr<int64_t>());

  at::Tensor mask_cpu = mask.to(at::kCPU);
  unsigned long long* mask_host =
      (unsigned long long*)mask_cpu.data_ptr<int64_t>();

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  at::Tensor keep_t =
      at::zeros({boxes_num}, boxes.options().dtype(at::kBool).device(at::kCPU));
  bool* keep = keep_t.data_ptr<bool>();

  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep[i] = true;
      // set every overlap box with bit 1 in remv
      unsigned long long* p = mask_host + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return order_t.masked_select(keep_t.to(at::kCUDA));
}
