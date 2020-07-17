#include "bbox_overlaps_cuda_kernel.cuh"
#include "parrots_cuda_helper.hpp"

void BBoxOverlapsCUDAKernelLauncher(const DArrayLite bboxes1,
                                    const DArrayLite bboxes2, DArrayLite ious,
                                    const int mode, const bool aligned,
                                    const int offset, cudaStream_t stream) {
  int output_size = ious.size();
  int num_bbox1 = bboxes1.dim(0);
  int num_bbox2 = bboxes2.dim(0);

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      bboxes1.elemType().prim(), ([&] {
        bbox_overlaps_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                bboxes1.ptr<scalar_t>(), bboxes2.ptr<scalar_t>(),
                ious.ptr<scalar_t>(), num_bbox1, num_bbox2, mode, aligned,
                offset);
      }));

  PARROTS_CUDA_CHECK(cudaGetLastError());
}
