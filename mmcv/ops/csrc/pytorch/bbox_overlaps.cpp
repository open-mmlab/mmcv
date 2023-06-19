// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiScalar;
using dipu::diopi_helper::toDiopiTensorHandle;
#endif

void bbox_overlaps_impl(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                        const int mode, const bool aligned, const int offset) {
  DISPATCH_DEVICE_IMPL(bbox_overlaps_impl, bboxes1, bboxes2, ious, mode,
                       aligned, offset);
}

#ifdef MMCV_WITH_DIOPI
void bbox_overlaps_diopi(const Tensor bboxes1, const Tensor bboxes2,
                         Tensor ious, const int mode, const bool aligned,
                         const int offset) {
  auto bboxes1_p = toDiopiTensorHandle(bboxes1);
  diopiDevice_t device;
  diopiGetTensorDevice(bboxes1_p, &device);
  if (device == diopi_host) {
    bbox_overlaps_impl(bboxes1, bboxes2, ious, mode, aligned, offset);
    return;
  }
  diopiContext ctx(dipu::getCurrentDIPUStream().rawstream());
  diopiContextHandle_t ch = &ctx;
  auto bboxes2_p = toDiopiTensorHandle(bboxes2);
  auto ious_p = toDiopiTensorHandle(ious);
  bool is_mock_cuda = bboxes1.device().type() == c10::DeviceType::PrivateUse1;
  if (is_mock_cuda &&
      reinterpret_cast<void *>(diopiBboxOverlapsMmcv) != nullptr) {
    auto ret = diopiBboxOverlapsMmcv(ch, ious_p, bboxes1_p, bboxes2_p, mode,
                                     offset, aligned);
    if (ret == diopiSuccess) return;
  }
  LOG(WARNING) << "Fallback to cpu: mmcv ext op bbox_overlaps";
  auto bboxes1_cpu = bboxes1.cpu();
  auto bboxes2_cpu = bboxes2.cpu();
  auto ious_cpu = ious.cpu();
  bbox_overlaps_impl(bboxes1_cpu, bboxes2_cpu, ious_cpu, mode, aligned, offset);
  ious.copy_(ious_cpu);
}
#endif

void bbox_overlaps(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                   const int mode, const bool aligned, const int offset) {
#ifdef MMCV_WITH_DIOPI
  bbox_overlaps_diopi(bboxes1, bboxes2, ious, mode, aligned, offset);
#else
  bbox_overlaps_impl(bboxes1, bboxes2, ious, mode, aligned, offset);
#endif
}
