// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated.h
#include "parrots_cpp_helper.hpp"

void box_iou_rotated_cpu_launcher(const DArrayLite boxes1,
                                  const DArrayLite boxes2, DArrayLite ious,
                                  const int mode_flag, const bool aligned);

void box_iou_rotated_cuda_launcher(const DArrayLite boxes1,
                                   const DArrayLite boxes2, DArrayLite ious,
                                   const int mode_flag, const bool aligned,
                                   cudaStream_t stream);

void box_iou_rotated_cpu(HostContext& ctx, const SSElement& attr,
                         const OperatorBase::in_list_t& ins,
                         OperatorBase::out_list_t& outs) {
  const auto& boxes1 = ins[0];
  const auto& boxes2 = ins[1];

  bool aligned;
  int mode_flag;
  SSAttrs(attr)
      .get<bool>("aligned", aligned)
      .get<int>("mode_flag", mode_flag)
      .done();
  auto& ious = outs[0];
  box_iou_rotated_cpu_launcher(boxes1, boxes2, ious, mode_flag, aligned);
}

void box_iou_rotated_cuda(CudaContext& ctx, const SSElement& attr,
                          const OperatorBase::in_list_t& ins,
                          OperatorBase::out_list_t& outs) {
  const auto& boxes1 = ins[0];
  const auto& boxes2 = ins[1];

  bool aligned;
  int mode_flag;
  SSAttrs(attr)
      .get<bool>("aligned", aligned)
      .get<int>("mode_flag", mode_flag)
      .done();

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  auto& ious = outs[0];
  box_iou_rotated_cuda_launcher(boxes1, boxes2, ious, mode_flag, aligned,
                                stream);
}

PARROTS_EXTENSION_REGISTER(box_iou_rotated)
    .attr("aligned")
    .attr("mode_flag")
    .input(2)
    .output(1)
    .apply(box_iou_rotated_cpu)
#ifdef PARROTS_USE_CUDA
    .apply(box_iou_rotated_cuda)
#endif
    .done();
