#include "pytorch_cpp_helper.hpp"

int BBoxOverlapsCUDAKernelLauncher(
    const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
    const int mode, const bool aligned, const int offset);

int bbox_overlaps_cuda(
    const Tensor bboxes1,
    const Tensor bboxes2,
    Tensor ious,
    const int mode,
    const bool aligned,
    const int offset) {
    return BBoxOverlapsCUDAKernelLauncher(
               bboxes1, bboxes2, ious, mode, aligned, offset);
}

int bbox_overlaps(
    const Tensor bboxes1,
    const Tensor bboxes2,
    Tensor ious,
    const int mode,
    const bool aligned,
    const int offset) {

    if (bboxes1.device().is_cuda()){
        CHECK_CUDA_INPUT(bboxes1);
        CHECK_CUDA_INPUT(bboxes2);
        CHECK_CUDA_INPUT(ious);

        return bbox_overlaps_cuda(bboxes1, bboxes2, ious, mode, aligned, offset);
    }
    return 0;
}
