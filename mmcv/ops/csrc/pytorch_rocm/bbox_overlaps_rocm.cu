#include "hip/hip_runtime.h"
#include "bbox_overlaps_cuda_kernel.cuh"
#include "pytorch_rocm_helper.hpp"

void BBoxOverlapsCUDAKernelLauncher(const Tensor bboxes1, const Tensor bboxes2,
                                    Tensor ious, const int mode,
                                    const bool aligned, const int offset) {
  int output_size = ious.numel();
  int num_bbox1 = bboxes1.size(0);
  int num_bbox2 = bboxes2.size(0);

  at::hip::HIPGuard device_guard(bboxes1.device());
  hipStream_t stream = at::hip::getCurrentHIPStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      bboxes1.scalar_type(), "bbox_overlaps_cuda_kernel", ([&] {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(bbox_overlaps_cuda_kernel<scalar_t>), dim3(GET_BLOCKS(output_size)), dim3(THREADS_PER_BLOCK), 0, stream, 
                bboxes1.data_ptr<scalar_t>(), bboxes2.data_ptr<scalar_t>(),
                ious.data_ptr<scalar_t>(), num_bbox1, num_bbox2, mode, aligned,
                offset);
      }));
  AT_CUDA_CHECK(hipGetLastError());
}
