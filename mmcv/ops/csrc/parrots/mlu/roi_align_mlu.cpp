// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/darray/darraymath.hpp>
#include <parrots/foundation/darrayutil.hpp>
#include <parrots_mlu_helper.hpp>
using namespace parrots;

#define USE_CPU_ROI_ALIGN

#ifdef USE_CPU_ROI_ALIGN
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

#endif  // USE_CPU_ROI_ALIGN

#ifdef PARROTS_USE_CAMB

void KernelRoiAlign(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                    cnrtQueue_t queue, const cnrtDataType_t d_type,
                    const void* input, const void* rois, const int channels,
                    const bool aligned, const int pooled_height,
                    const int pooled_width, const int input_height,
                    const int input_width, const int sampling_ratio,
                    const float spatial_scale, const int num_rois,
                    void* output);

void KernelRoiAlignBackward(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                            cnrtQueue_t queue, const cnrtDataType_t dtype,
                            const void* grads, const void* boxes,
                            void* grads_image, const int boxes_num,
                            const int hi, const int wi, const int c,
                            const int no, const int ho, const int wo,
                            const float spatial_scale, const int sampling_ratio,
                            const bool aligned);

namespace roi_align_forward {

void ROIAlignForwardMLUKernelLauncher(CambContext& ctx, const DArrayLite& input,
                                      const DArrayLite& rois,
                                      DArrayLite& output, DArrayLite& argmax_y,
                                      DArrayLite& argmax_x, int aligned_height,
                                      int aligned_width, float spatial_scale,
                                      int sampling_ratio, int pool_mode,
                                      bool aligned) {
  // params check
  PARROTS_CHECKARGS(input.elemType() == Prim::Float32 ||
                    input.elemType() == Prim::Float16)
      << "input type should be Float or Half, bug got " << input.elemType();
  PARROTS_CHECKARGS(rois.elemType() == input.elemType())
      << "rois should have the same type as input, bug got" << rois.elemType()
      << input.elemType();
  PARROTS_CHECKARGS(input.ndims() == 4)
      << "input should be a 4d tensor, got " << input.ndims() << "D";
  PARROTS_CHECKARGS(rois.ndims() == 2)
      << "rois should be a 2d tensor, got " << rois.ndims() << "D";
  PARROTS_CHECKARGS(pool_mode == 1)
      << "pool_mode only suppurts 'avg' currently";

  const auto num_rois = rois.dim(0);
  const auto channels = input.dim(1);
  const int height = input.dim(2);
  const int width = input.dim(3);
  const auto mem_format = MemoryFormat::ChannelsLast;

  const DArrayLite* input_ptr = &input;
  const DArrayLite* output_ptr = &output;
  DArrayLite input_tmp;
  DArrayLite output_tmp;
  const auto input_memformat = input.spec().probableMemoryFormat();
  if (input_memformat != MemoryFormat::ChannelsLast) {
    input_tmp = ctx.createDArrayLite(
        input.spec().duplicate(MemoryFormat::ChannelsLast));
    copy(ctx, input_tmp, input);
    input_ptr = &input_tmp;
  }
  if (output.spec().probableMemoryFormat() != MemoryFormat::ChannelsLast ||
      output.size() <= 0) {
    output_tmp = ctx.createDArrayLite(
        input.spec()
            .withShape(
                DArrayShape(num_rois, channels, aligned_height, aligned_width))
            .duplicate(MemoryFormat::ChannelsLast));
    output_ptr = &output_tmp;
  }

  auto queue = getStreamNative<CambDevice>(ctx.getStream());

  cnrtDim3_t k_dim;
  cnrtJobType_t k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim.x = getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim.y = getDeviceAttr(cnrtAttrClusterCount);
  k_dim.z = 1;
  cnrtDataType_t data_type = getCnrtDataType(input.elemType());

  KernelRoiAlign(k_dim, k_type, queue, data_type, (void*)input_ptr->data(),
                 rois.data(), channels, aligned, aligned_height, aligned_width,
                 height, width, sampling_ratio, spatial_scale, num_rois,
                 (void*)output_ptr->data());
  if (output_tmp.size() > 0) {
    if (output.size() <= 0) {
      if (input_memformat == MemoryFormat::Contiguous) {
        output = ctx.createDArrayLite(output_tmp);
      } else {
        output = ctx.cloneDArrayLite(output_tmp);
      }
    } else {
      copy(ctx, output, output_tmp);
    }
  }
}

void roi_align_forward_camb_parrots(CambContext& ctx, const SSElement& attr,
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

  const auto& input = ins[0];
  const auto& rois = ins[1];
  auto& output = outs[0];
  auto& argmax_y = outs[1];
  auto& argmax_x = outs[2];
  ROIAlignForwardMLUKernelLauncher(ctx, input, rois, output, argmax_y, argmax_x,
                                   aligned_height, aligned_width, spatial_scale,
                                   sampling_ratio, pool_mode, aligned);
}

}  //  namespace roi_align_forward

namespace roi_align_backward {
static int nearestPower2(int x) {
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  return x;
}

void ROIAlignBackwardMLUKernelLauncher(
    CambContext& ctx, const DArrayLite& grad, const DArrayLite& rois,
    const DArrayLite& argmax_y, const DArrayLite& argmax_x,
    DArrayLite& grad_input, int aligned_height, int aligned_width,
    float spatial_scale, int sampling_ratio, int pool_mode, bool aligned) {
  // params check
  PARROTS_CHECKARGS(grad.elemType() == Prim::Float16 ||
                    grad.elemType() == Prim::Float32)
      << "grad type should be Float or Half, got " << grad.elemType();
  PARROTS_CHECKARGS(rois.elemType() == grad.elemType())
      << "rois should have the same type as grad";
  PARROTS_CHECKARGS(grad.ndims() == 4)
      << "grad should be a 4D tensor, got " << grad.ndims() << "D";
  PARROTS_CHECKARGS(rois.ndims() == 2)
      << "rois should be a 2D tensor, got " << rois.ndims() << "D";
  PARROTS_CHECKARGS(pool_mode == 1)
      << "pool_mode only suppurts 'avg' currently";

  const int batch_size = grad_input.dim(0);
  const int channels = grad_input.dim(1);
  const int height = grad_input.dim(2);
  const int width = grad_input.dim(3);

  const int boxes_num = rois.dim(0);
  const int c = grad.dim(1);
  const int hi = grad.dim(2);
  const int wi = grad.dim(3);

  const int no = grad_input.dim(0);
  const int ho = grad_input.dim(2);
  const int wo = grad_input.dim(3);

  const DArrayLite* grad_ptr = &grad;
  DArrayLite grad_tmp;
  if (grad.spec().probableMemoryFormat() != MemoryFormat::ChannelsLast) {
    grad_tmp =
        ctx.createDArrayLite(grad.spec().duplicate(MemoryFormat::ChannelsLast));
    copy(ctx, grad_tmp, grad);
    grad_ptr = &grad_tmp;
  }

  DArrayLite* grad_input_ptr = &grad_input;
  DArrayLite grad_input_;
  if (grad.spec().probableMemoryFormat() != MemoryFormat::ChannelsLast ||
      grad_input.size() <= 0) {
    grad_input_ = ctx.createDArrayLite(grad_input.spec().withShape(
        DArrayShape(batch_size, channels, height, width),
        MemoryFormat::ChannelsLast));
    grad_input_ptr = &grad_input_;
  }

  PARROTS_CALLCNRT(
      cnrtMemset(grad_input_ptr->data(), 0, grad_input_ptr->nbytes()));

  cnrtJobType_t k_type = CNRT_FUNC_TYPE_UNION1;
  int need_core = nearestPower2(boxes_num);
  int union_number = getDeviceAttr(cnrtAttrClusterCount);
  uint32_t dim_x = getDeviceAttr(cnrtAttrMcorePerCluster);
  uint32_t dim_y = (need_core - 1) / dim_x + 1;
  dim_y = (dim_y > union_number) ? union_number : dim_y;
  cnrtDim3_t k_dim = {dim_x, dim_y, 1};
  cnrtDataType_t k_dtype = getCnrtDataType(grad.elemType());
  auto queue = getStreamNative<CambDevice>(ctx.getStream());

  KernelRoiAlignBackward(
      k_dim, k_type, queue, k_dtype, const_cast<void*>(grad_ptr->data()),
      const_cast<void*>(rois.data()), grad_input_ptr->data(), boxes_num, hi, wi,
      c, no, ho, wo, spatial_scale, sampling_ratio, aligned);
  if (grad_input_.size() > 0) {
    grad_input = ctx.createDArrayLite(grad_input_);
  }
}

void roi_align_backward_camb_parrots(CambContext& ctx, const SSElement& attr,
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
  const auto& grad_output = ins[0];
  const auto& rois = ins[1];
  const auto& argmax_y = ins[2];
  const auto& argmax_x = ins[3];
  auto& grad_input = outs[0];
  ROIAlignBackwardMLUKernelLauncher(
      ctx, grad_output, rois, argmax_y, argmax_x, grad_input, aligned_height,
      aligned_width, spatial_scale, sampling_ratio, pool_mode, aligned);
}

}  //  namespace roi_align_backward

#endif  //  PARROTS_USE_CAMB

PARROTS_EXTENSION_REGISTER(roi_align_forward)
    .attr("aligned_height")
    .attr("aligned_width")
    .attr("spatial_scale")
    .attr("sampling_ratio")
    .attr("pool_mode")
    .attr("aligned")
    .input(2)
    .output(3)
#ifdef USE_CPU_ROI_ALIGN
    .apply(roi_align_forward_cpu_parrots)
#endif
#ifdef PARROTS_USE_CAMB
    .apply(roi_align_forward::roi_align_forward_camb_parrots)
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
#ifdef USE_CPU_ROI_ALIGN
    .apply(roi_align_backward_cpu_parrots)
#endif
#ifdef PARROTS_USE_CAMB
    .apply(roi_align_backward::roi_align_backward_camb_parrots)
#endif
    .done();
