
#include <parrots/compute/aten.hpp>
#include <parrots/darray/darraymath.hpp>
#include <parrots/foundation/darrayutil.hpp>

#include "parrots_roi_align.h"

using namespace parrots;

using at::Tensor;
void ROIAlignForwardCPULauncher(Tensor input, Tensor rois, Tensor output,
                                Tensor argmax_y, Tensor argmax_x,
                                int aligned_height, int aligned_width,
                                float spatial_scale, int sampling_ratio,
                                int pool_mode, bool aligned);

void ROIAlignBackwardCPULauncher(Tensor grad_output, Tensor rois,
                                 Tensor argmax_y, Tensor argmax_x,
                                 Tensor grad_input, int aligned_height,
                                 int aligned_width, float spatial_scale,
                                 int sampling_ratio, int pool_mode,
                                 bool aligned);

void roi_align_forward_cpu_parrots(HostContext& ctx, const SSElement& attr,
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

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& rois = buildATensor(ctx, ins[1]);
  auto output = buildATensor(ctx, outs[0]);
  auto argmax_y = buildATensor(ctx, outs[1]);
  auto argmax_x = buildATensor(ctx, outs[2]);

  ROIAlignForwardCPULauncher(input, rois, output, argmax_y, argmax_x,
                             aligned_height, aligned_width, spatial_scale,
                             sampling_ratio, pool_mode, aligned);
}

void roi_align_backward_cpu_parrots(HostContext& ctx, const SSElement& attr,
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
  const auto& grad_output = buildATensor(ctx, ins[0]);
  const auto& rois = buildATensor(ctx, ins[1]);
  const auto& argmax_y = buildATensor(ctx, ins[2]);
  const auto& argmax_x = buildATensor(ctx, ins[3]);
  auto grad_input = buildATensor(ctx, outs[0]);
  ROIAlignBackwardCPULauncher(grad_output, rois, argmax_y, argmax_x, grad_input,
                              aligned_height, aligned_width, spatial_scale,
                              sampling_ratio, pool_mode, aligned);
}

REGISTER_DEVICE_IMPL(roi_align_forward_impl, CPU, Arch::X86,
                     roi_align_forward_cpu_parrots);
REGISTER_DEVICE_IMPL(roi_align_backward_impl, CPU, Arch::X86,
                     roi_align_backward_cpu_parrots);
