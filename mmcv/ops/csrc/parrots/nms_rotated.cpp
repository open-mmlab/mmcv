// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include "parrots_cpp_helper.hpp"

DArrayLite nms_rotated_cuda(const DArrayLite dets, const DArrayLite scores,
                            const DArrayLite dets_sorted, float iou_threshold,
                            cudaStream_t stream, CudaContext& ctx);

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
void nms_rotated(CudaContext& ctx, const SSElement& attr,
                 const OperatorBase::in_list_t& ins,
                 OperatorBase::out_list_t& outs) {
  float iou_threshold;
  SSAttrs(attr).get<float>("iou_threshold", iou_threshold).done();

  const auto& dets = ins[0];
  const auto& scores = ins[1];
  const auto& dets_sorted = ins[2];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());

  outs[0] =
      nms_rotated_cuda(dets, scores, dets_sorted, iou_threshold, stream, ctx);
}

PARROTS_EXTENSION_REGISTER(nms_rotated)
    .attr("iou_threshold")
    .input(3)
    .output(1)
    .apply(nms_rotated)
    .done();
