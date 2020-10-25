// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include "parrots_cpp_helper.hpp"

DArrayLite box_iou_rotated_cuda(
    const DArrayLite boxes1,
    const DArrayLite boxes2,
    cudaStream_t stream,
    CudaContext& ctx);

void box_iou_rotated(CudaContext& ctx, const SSElement& attr,
                        const OperatorBase::in_list_t& ins,
                        OperatorBase::out_list_t& outs) {

  const auto& boxes1 = ins[0];
  const auto& boxes2 = ins[1];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  outs[0] = box_iou_rotated_cuda(boxes1, boxes2, stream, ctx);
}


PARROTS_EXTENSION_REGISTER(box_iou_rotated)
    .input(2)
    .output(1)
    .apply(box_iou_rotated)
    .done();
