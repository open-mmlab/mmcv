#include "parrots_cpp_helper.hpp"

void ROIPoolForwardCUDAKernelLauncher(const DArrayLite input,
                                      const DArrayLite rois, DArrayLite output,
                                      DArrayLite argmax, int pooled_height,
                                      int pooled_width, float spatial_scale,
                                      cudaStream_t stream);

void ROIPoolBackwardCUDAKernelLauncher(const DArrayLite grad_output,
                                       const DArrayLite rois,
                                       const DArrayLite argmax,
                                       DArrayLite grad_input, int pooled_height,
                                       int pooled_width, float spatial_scale,
                                       cudaStream_t stream);

void roi_pool_forward_cuda(CudaContext& ctx, const SSElement& attr,
                           const OperatorBase::in_list_t& ins,
                           OperatorBase::out_list_t& outs) {
  int pooled_height;
  int pooled_width;
  float spatial_scale;
  SSAttrs(attr)
      .get<int>("pooled_height", pooled_height)
      .get<int>("pooled_width", pooled_width)
      .get<float>("spatial_scale", spatial_scale)
      .done();

  const auto& input = ins[0];
  const auto& rois = ins[1];
  auto& output = outs[0];
  auto& argmax = outs[1];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  ROIPoolForwardCUDAKernelLauncher(input, rois, output, argmax, pooled_height,
                                   pooled_width, spatial_scale, stream);
}

void roi_pool_backward_cuda(CudaContext& ctx, const SSElement& attr,
                            const OperatorBase::in_list_t& ins,
                            OperatorBase::out_list_t& outs) {
  int pooled_height;
  int pooled_width;
  float spatial_scale;
  SSAttrs(attr)
      .get<int>("pooled_height", pooled_height)
      .get<int>("pooled_width", pooled_width)
      .get<float>("spatial_scale", spatial_scale)
      .done();

  const auto& grad_output = ins[0];
  const auto& rois = ins[1];
  const auto& argmax = ins[2];
  auto& grad_input = outs[0];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  ROIPoolBackwardCUDAKernelLauncher(grad_output, rois, argmax, grad_input,
                                    pooled_height, pooled_width, spatial_scale,
                                    stream);
}

PARROTS_EXTENSION_REGISTER(roi_pool_forward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("spatial_scale")
    .input(2)
    .output(2)
    .apply(roi_pool_forward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(roi_pool_backward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("spatial_scale")
    .input(3)
    .output(1)
    .apply(roi_pool_backward_cuda)
    .done();
