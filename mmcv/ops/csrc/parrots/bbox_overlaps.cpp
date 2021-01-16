#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void BBoxOverlapsCUDAKernelLauncher(const Tensor bboxes1, const Tensor bboxes2,
                                    Tensor ious, const int mode,
                                    const bool aligned, const int offset);

void bbox_overlaps_cuda(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                        const int mode, const bool aligned, const int offset) {
  BBoxOverlapsCUDAKernelLauncher(bboxes1, bboxes2, ious, mode, aligned, offset);
}
#endif

void bbox_overlaps(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                   const int mode, const bool aligned, const int offset) {
  if (bboxes1.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(bboxes1);
    CHECK_CUDA_INPUT(bboxes2);
    CHECK_CUDA_INPUT(ious);

    bbox_overlaps_cuda(bboxes1, bboxes2, ious, mode, aligned, offset);
#else
    AT_ERROR("bbox_overlaps is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("bbox_overlaps is not implemented on CPU");
  }
}
