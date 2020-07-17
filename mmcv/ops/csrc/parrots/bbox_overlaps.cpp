#include "parrots_cpp_helper.hpp"

void BBoxOverlapsCUDAKernelLauncher(const DArrayLite bboxes1,
                                    const DArrayLite bboxes2, DArrayLite ious,
                                    const int mode, const bool aligned,
                                    const int offset, cudaStream_t stream);

void bbox_overlaps_cuda(CudaContext& ctx, const SSElement& attr,
                        const OperatorBase::in_list_t& ins,
                        OperatorBase::out_list_t& outs) {
  int mode, offset;
  bool aligned;
  SSAttrs(attr)
      .get<int>("mode", mode)
      .get<bool>("aligned", aligned)
      .get<int>("offset", offset)
      .done();

  const auto& bboxes1 = ins[0];
  const auto& bboxes2 = ins[1];

  auto& ious = outs[0];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  BBoxOverlapsCUDAKernelLauncher(bboxes1, bboxes2, ious, mode, aligned, offset,
                                 stream);
}

PARROTS_EXTENSION_REGISTER(bbox_overlaps)
    .attr("mode")
    .attr("aligned")
    .attr("offset")
    .input(2)
    .output(1)
    .apply(bbox_overlaps_cuda)
    .done();
