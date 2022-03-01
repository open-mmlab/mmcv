#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void BBoxOverlapsMLUKernelLauncher(const Tensor bboxes1, const Tensor bboxes2,
                                   Tensor ious, const int mode,
                                   const bool aligned, const int offset);

void bbox_overlaps_mlu(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                       const int mode, const bool aligned, const int offset) {
  BBoxOverlapsMLUKernelLauncher(bboxes1, bboxes2, ious, mode, aligned, offset);
}

void bbox_overlaps_impl(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                        const int mode, const bool aligned, const int offset);
REGISTER_DEVICE_IMPL(bbox_overlaps_impl, MLU, bbox_overlaps_mlu);

