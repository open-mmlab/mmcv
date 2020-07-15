#include "parrots_cpp_helper.hpp"

void DeformRoIPoolForwardCUDAKernelLauncher(
    const DArrayLite input, const DArrayLite rois, const DArrayLite offset,
    DArrayLite output, int pooled_height, int pooled_width, float spatial_scale,
    int sampling_ratio, float gamma, cudaStream_t stream);

void DeformRoIPoolBackwardCUDAKernelLauncher(
    const DArrayLite grad_output, const DArrayLite input, const DArrayLite rois,
    const DArrayLite offset, DArrayLite grad_input, DArrayLite grad_offset,
    int pooled_height, int pooled_width, float spatial_scale,
    int sampling_ratio, float gamma, cudaStream_t stream);

void deform_roi_pool_forward_cuda(CudaContext& ctx, const SSElement& attr,
                                  const OperatorBase::in_list_t& ins,
                                  OperatorBase::out_list_t& outs) {
  int pooled_height;
  int pooled_width;
  float spatial_scale;
  int sampling_ratio;
  float gamma;
  SSAttrs(attr)
      .get<int>("pooled_height", pooled_height)
      .get<int>("pooled_width", pooled_width)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("sampling_ratio", sampling_ratio)
      .get<float>("gamma", gamma)
      .done();

  const auto& input = ins[0];
  const auto& rois = ins[1];
  const auto& offset = ins[2];

  auto& output = outs[0];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  DeformRoIPoolForwardCUDAKernelLauncher(
      input, rois, offset, output, pooled_height, pooled_width, spatial_scale,
      sampling_ratio, gamma, stream);
}

void deform_roi_pool_backward_cuda(CudaContext& ctx, const SSElement& attr,
                                   const OperatorBase::in_list_t& ins,
                                   OperatorBase::out_list_t& outs) {
  int pooled_height;
  int pooled_width;
  float spatial_scale;
  int sampling_ratio;
  float gamma;

  SSAttrs(attr)
      .get<int>("pooled_height", pooled_height)
      .get<int>("pooled_width", pooled_width)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("sampling_ratio", sampling_ratio)
      .get<float>("gamma", gamma)
      .done();

  const auto& grad_output = ins[0];
  const auto& input = ins[1];
  const auto& rois = ins[2];
  const auto& offset = ins[3];

  auto& grad_input = outs[0];
  auto& grad_offset = outs[1];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  DeformRoIPoolBackwardCUDAKernelLauncher(
      grad_output, input, rois, offset, grad_input, grad_offset, pooled_height,
      pooled_width, spatial_scale, sampling_ratio, gamma, stream);
}

PARROTS_EXTENSION_REGISTER(deform_roi_pool_forward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("spatial_scale")
    .attr("sampling_ratio")
    .attr("gamma")
    .input(3)
    .output(1)
    .apply(deform_roi_pool_forward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(deform_roi_pool_backward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("spatial_scale")
    .attr("sampling_ratio")
    .attr("gamma")
    .input(4)
    .output(2)
    .apply(deform_roi_pool_backward_cuda)
    .done();
