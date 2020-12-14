// Copyright (c) 2018, SenseTime.
#include "parrots_cpp_helper.hpp"

void ROIAlignForwardCPULauncher(DArrayLite input, DArrayLite rois,
                                DArrayLite output, DArrayLite argmax_y,
                                DArrayLite argmax_x, int aligned_height,
                                int aligned_width, float spatial_scale,
                                int sampling_ratio, int pool_mode,
                                bool aligned);

void ROIAlignBackwardCPULauncher(DArrayLite grad_output, DArrayLite rois,
                                 DArrayLite argmax_y, DArrayLite argmax_x,
                                 DArrayLite grad_input, int aligned_height,
                                 int aligned_width, float spatial_scale,
                                 int sampling_ratio, int pool_mode,
                                 bool aligned);

void ROIAlignForwardCUDAKernelLauncher(DArrayLite input, DArrayLite rois,
                                       DArrayLite output, DArrayLite argmax_y,
                                       DArrayLite argmax_x, int aligned_height,
                                       int aligned_width, float spatial_scale,
                                       int sampling_ratio, int pool_mode,
                                       bool aligned, cudaStream_t stream);

void ROIAlignBackwardCUDAKernelLauncher(
    DArrayLite grad_output, DArrayLite rois, DArrayLite argmax_y,
    DArrayLite argmax_x, DArrayLite grad_input, int aligned_height,
    int aligned_width, float spatial_scale, int sampling_ratio, int pool_mode,
    bool aligned, cudaStream_t stream);

void roi_align_forward_cpu(HostContext& ctx, const SSElement& attr,
                           const OperatorBase::in_list_t& ins,
                           OperatorBase::out_list_t& outs) {
  int aligned_height;
  int aligned_width;
  float spatial_scale;
  int sampling_ratio;
  int pool_mode;
  bool aligned;
  SSAttrs(attr)
      .get<int>("aligned_height", aligned_height)
      .get<int>("aligned_width", aligned_width)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("sampling_ratio", sampling_ratio)
      .get<int>("pool_mode", pool_mode)
      .get<bool>("aligned", aligned)
      .done();

  auto& input = ins[0];
  auto& rois = ins[1];
  auto& output = outs[0];
  auto& argmax_y = outs[1];
  auto& argmax_x = outs[2];

  ROIAlignForwardCPULauncher(input, rois, output, argmax_y, argmax_x,
                             aligned_height, aligned_width, spatial_scale,
                             sampling_ratio, pool_mode, aligned);
}

void roi_align_backward_cpu(HostContext& ctx, const SSElement& attr,
                            const OperatorBase::in_list_t& ins,
                            OperatorBase::out_list_t& outs) {
  int aligned_height;
  int aligned_width;
  float spatial_scale;
  int sampling_ratio;
  int pool_mode;
  bool aligned;
  SSAttrs(attr)
      .get<int>("aligned_height", aligned_height)
      .get<int>("aligned_width", aligned_width)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("sampling_ratio", sampling_ratio)
      .get<int>("pool_mode", pool_mode)
      .get<bool>("aligned", aligned)
      .done();

  auto& grad_output = ins[0];
  auto& rois = ins[1];
  auto& argmax_y = ins[2];
  auto& argmax_x = ins[3];
  auto& grad_input = outs[0];

  ROIAlignBackwardCPULauncher(grad_output, rois, argmax_y, argmax_x, grad_input,
                              aligned_height, aligned_width, spatial_scale,
                              sampling_ratio, pool_mode, aligned);
}

void roi_align_forward_cuda(CudaContext& ctx, const SSElement& attr,
                            const OperatorBase::in_list_t& ins,
                            OperatorBase::out_list_t& outs) {
  int aligned_height;
  int aligned_width;
  float spatial_scale;
  int sampling_ratio;
  int pool_mode;
  bool aligned;
  SSAttrs(attr)
      .get<int>("aligned_height", aligned_height)
      .get<int>("aligned_width", aligned_width)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("sampling_ratio", sampling_ratio)
      .get<int>("pool_mode", pool_mode)
      .get<bool>("aligned", aligned)
      .done();

  auto& input = ins[0];
  auto& rois = ins[1];
  auto& output = outs[0];
  auto& argmax_y = outs[1];
  auto& argmax_x = outs[2];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  ROIAlignForwardCUDAKernelLauncher(
      input, rois, output, argmax_y, argmax_x, aligned_height, aligned_width,
      spatial_scale, sampling_ratio, pool_mode, aligned, stream);
}

void roi_align_backward_cuda(CudaContext& ctx, const SSElement& attr,
                             const OperatorBase::in_list_t& ins,
                             OperatorBase::out_list_t& outs) {
  int aligned_height;
  int aligned_width;
  float spatial_scale;
  int sampling_ratio;
  int pool_mode;
  bool aligned;
  SSAttrs(attr)
      .get<int>("aligned_height", aligned_height)
      .get<int>("aligned_width", aligned_width)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("sampling_ratio", sampling_ratio)
      .get<int>("pool_mode", pool_mode)
      .get<bool>("aligned", aligned)
      .done();

  auto& grad_output = ins[0];
  auto& rois = ins[1];
  auto& argmax_y = ins[2];
  auto& argmax_x = ins[3];
  auto& grad_input = outs[0];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  ROIAlignBackwardCUDAKernelLauncher(
      grad_output, rois, argmax_y, argmax_x, grad_input, aligned_height,
      aligned_width, spatial_scale, sampling_ratio, pool_mode, aligned, stream);
}

PARROTS_EXTENSION_REGISTER(roi_align_forward)
    .attr("aligned_height")
    .attr("aligned_width")
    .attr("spatial_scale")
    .attr("sampling_ratio")
    .attr("pool_mode")
    .attr("aligned")
    .input(2)
    .output(3)
    .apply(roi_align_forward_cpu)
#ifdef PARROTS_USE_CUDA
    .apply(roi_align_forward_cuda)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(roi_align_backward)
    .attr("aligned_height")
    .attr("aligned_width")
    .attr("spatial_scale")
    .attr("sampling_ratio")
    .attr("pool_mode")
    .attr("aligned")
    .input(4)
    .output(1)
    .apply(roi_align_backward_cpu)
#ifdef PARROTS_USE_CUDA
    .apply(roi_align_backward_cuda)
#endif
    .done();
