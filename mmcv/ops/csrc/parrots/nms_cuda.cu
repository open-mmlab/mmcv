#include "nms_cuda_kernel.cuh"
#include "parrots_cuda_helper.hpp"

DArrayLite NMSCUDAKernelLauncher(const DArrayLite boxes_sorted,
                                 const DArrayLite order, const DArrayLite areas,
                                 float iou_threshold, int offset,
                                 CudaContext& ctx, cudaStream_t stream) {
  size_t boxes_num = boxes_sorted.dim(0);

  if (boxes_sorted.size() == 0) {
    auto select = ctx.createDArrayLite(DArraySpec::array(Prim::Int64, 0));
    return select;
  }

  const size_t col_blocks = DIVUP(boxes_num, threadsPerBlock);
  auto mask = ctx.createDArrayLite(
      DArraySpec::array(Prim::Int64, DArrayShape(boxes_num, col_blocks)));
  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);
  PARROTS_CUDA_CHECK(cudaGetLastError());
  nms_cuda<<<blocks, threads, 0, stream>>>(
      boxes_num, iou_threshold, offset, boxes_sorted.ptr<float>(),
      (unsigned long long*)mask.ptr<int64_t>());
  PARROTS_CUDA_CHECK(cudaGetLastError());

  auto mask_cpu = ctx.createDArrayLite(mask, getHostProxy());
  auto mask_host = mask_cpu.ptr<int64_t>();

  auto remv = ctx.createDArrayLite(DArraySpec::array(Prim::Int64, col_blocks),
                                   getHostProxy());
  remv.setZeros(syncStream());
  auto remv_ptr = remv.ptr<int64_t>();

  auto keep_t = ctx.createDArrayLite(DArraySpec::array(Prim::Uint8, boxes_num),
                                     getHostProxy());
  keep_t.setZeros(syncStream());
  auto keep = keep_t.ptr<uint8_t>();

  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv_ptr[nblock] & (1ULL << inblock))) {
      keep[i] = 1;
      int64_t* p = mask_host + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv_ptr[j] |= p[j];
      }
    }
  }

  auto keep_cuda = ctx.createDArrayLite(keep_t, ctx.getProxy());
  PARROTS_CUDA_CHECK(cudaGetLastError());
  return keep_cuda;
}
