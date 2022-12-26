// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "deform_roi_pool_pytorch.h"
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <parrots/diopi.hpp>

using namespace parrots;

#ifdef MMCV_WITH_CUDA
/*void deform_roi_pool_forward_cuda(Tensor input, Tensor rois, Tensor offset,
 *                                  Tensor output, int pooled_height,
 *                                  int pooled_width, float spatial_scale,
 *                                  int sampling_ratio, float gamma);
 */
void deform_roi_pool_forward_cuda_parrots(CudaContext& ctx,
                                          const SSElement& attr,
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

  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto input = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));
  auto rois = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[1]));
  auto offset = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[2]));

  auto output = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);
  PARROTS_CALLDIOPI(diopiDeformRoiPool(ch, input, rois, offset, output, pooled_height,
                               pooled_width, spatial_scale, sampling_ratio,
                               gamma));
}

/*void deform_roi_pool_backward_cuda(Tensor grad_output, Tensor input,
 *                                   Tensor rois, Tensor offset,
 *                                   Tensor grad_input, Tensor grad_offset,
 *                                   int pooled_height, int pooled_width,
 *                                   float spatial_scale, int sampling_ratio,
 *                                   float gamma);
 */
void deform_roi_pool_backward_cuda_parrots(CudaContext& ctx,
                                           const SSElement& attr,
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

  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto grad_output = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));
  auto input = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[1]));
  auto rois = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[2]));
  auto offset = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[3]));

  auto grad_input = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);
  auto grad_offset = reinterpret_cast<diopiTensorHandle_t>(&outs[1]);

  PARROTS_CALLDIOPI(diopiDeformRoiPoolBackward(ch, grad_output, input, rois, offset, grad_input,
                                grad_offset, pooled_height, pooled_width,
                                spatial_scale, sampling_ratio, gamma));
}

PARROTS_EXTENSION_REGISTER(deform_roi_pool_forward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("spatial_scale")
    .attr("sampling_ratio")
    .attr("gamma")
    .input(3)
    .output(1)
    .apply(deform_roi_pool_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(deform_roi_pool_backward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("spatial_scale")
    .attr("sampling_ratio")
    .attr("gamma")
    .input(4)
    .output(2)
    .apply(deform_roi_pool_backward_cuda_parrots)
    .done();
#endif
