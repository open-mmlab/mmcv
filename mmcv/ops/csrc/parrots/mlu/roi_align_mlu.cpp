// Copyright (c) OpenMMLab. All rights reserved
#include<parrots_mlu_helper.hpp>
#include <parrots/compute/aten.hpp>

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

#endif // USE_CPU_ROI_ALIGN

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
                            cnrtQueue_t queue, cnrtDataType_t dtype,
                            void* grads, void* boxes, void* grads_image,
                            int boxes_num, int hi, int wi, int c, int no,
                            int ho, int wo, float spatial_scale,
                            int sampling_ratio, bool aligned);

namespace roi_align_forward {
static void policyFunc(cnrtQueue_t queue, cnrtDim3_t* k_dim,
                       cnrtFunctionType_t* k_type) {
  k_dim->x = getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim->y = getDeviceAttr(cnrtAttrClusterCount);
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_UNION1;
}

void ROIAlignForwardMLUKernelLauncher(CambContext& ctx,const DArrayLite& input,
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

  auto num_rois = rois.dim(0);
  auto channels = input.dim(1);
  int height = input.dim(2);
  int width = input.dim(3);
  if (output.size() == 0) {
    output = ctx.createDArrayLite(input.spec().withShape(
        DArrayShape(num_rois, channels, aligned_height, aligned_width)));
  }

  DArrayLite output_tmp = output.view(input.spec().withShape(
        DArrayShape(num_rois, channels, aligned_height, aligned_width),
        parrots::MemoryFormat::ChannelsLast));

  auto queue = getStreamNative<CambDevice>(ctx.getStream());

  cnrtDim3_t k_dim;
  cnrtJobType_t k_type;
  policyFunc(queue, &k_dim, &k_type);
  cnrtDataType_t data_type = getCnrtDataType(input.elemType());


  KernelRoiAlign(k_dim, k_type, queue, data_type, input.data(), rois.data(), channels, aligned,
                 aligned_height, aligned_width, height, width, sampling_ratio,
                 spatial_scale, num_rois, output_tmp.data());
}

void KernelRoiAlign(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                    cnrtQueue_t queue, const cnrtDataType_t d_type,
                    const void* input, const void* rois, const int channels,
                    const bool aligned, const int pooled_height,
                    const int pooled_width, const int input_height,
                    const int input_width, const int sampling_ratio,
                    const float spatial_scale, const int num_rois,
                    void* output);

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
  ROIAlignForwardMLUKernelLauncher(ctx,
      input, rois, output, argmax_y, argmax_x, aligned_height, aligned_width,
      spatial_scale, sampling_ratio, pool_mode, aligned);
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

void ROIAlignBackwardMLUKernelLauncher(CambContext& ctx,
                                       const DArrayLite& grad,
                                       const DArrayLite&  rois,
                                       const DArrayLite&  argmax_y,
                                       const DArrayLite&  argmax_x,
                                       DArrayLite& grad_input,
                                       int aligned_height,
                                       int aligned_width,
                                       float spatial_scale,
                                       int sampling_ratio,
                                       int pool_mode,
                                       bool aligned) {
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

  int batch_size = grad_input.dim(0);
  int channels = grad_input.dim(1);
  int height = grad_input.dim(2);
  int width = grad_input.dim(3);

  int boxes_num = rois.dim(0);
  int c = grad.dim(1);
  int hi = grad.dim(2);
  int wi = grad.dim(3);

  int no = grad_input.dim(0);
  int ho = grad_input.dim(2);
  int wo = grad_input.dim(3);

  DArrayLite grad_input_ = grad_input.view(grad_input.spec().withShape(
        DArrayShape(batch_size, channels, height, width),
        parrots::MemoryFormat::ChannelsLast));

  cnrtJobType_t k_type = CNRT_FUNC_TYPE_UNION1;
  int need_core = nearestPower2(boxes_num);
  int union_number = getDeviceAttr(cnrtAttrClusterCount);
  uint32_t dim_x = getDeviceAttr(cnrtAttrMcorePerCluster);
  uint32_t dim_y = (need_core - 1) / dim_x + 1;
  dim_y = (dim_y > union_number) ? union_number : dim_y;
  cnrtDim3_t k_dim = {dim_x, dim_y, 1};
  cnrtDataType_t k_dtype = getCnrtDataType(grad.elemType());
  auto queue = getStreamNative<CambDevice>(ctx.getStream());

  KernelRoiAlignBackward(k_dim, k_type, queue, k_dtype, const_cast<void*>(grad.data()),
                         const_cast<void*>(rois.data()), grad_input_.data(), boxes_num, hi,
                         wi, c, no, ho, wo, spatial_scale, sampling_ratio,
                         aligned);

  /*

  grad_input.copy_(grad_input_);
  */
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

#endif //  PARROTS_USE_CAMB

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
