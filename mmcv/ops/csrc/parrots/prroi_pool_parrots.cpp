// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "prroi_pool_pytorch.h"
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <parrots/diopi.hpp>

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void prroi_pool_forward_cuda_parrots_diopi(CudaContext& ctx, const SSElement& attr,
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

  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto input = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));
  auto rois = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[1]));
  auto output = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);
  PARROTS_CALLDIOPI(diopiPrroiPool(ch, input, rois, output, pooled_height, pooled_width, spatial_scale));
}

void prroi_pool_backward_cuda_parrots_diopi(CudaContext& ctx, const SSElement& attr,
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

  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto grad_output = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));
  auto rois = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[1]));
  auto grad_input = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);
  PARROTS_CALLDIOPI(diopiPrroiPoolbackward(ch, grad_output, rois, grad_input, pooled_height,
                                           pooled_width, spatial_scale));
}

void prroi_pool_coor_backward_cuda_parrots_diopi(CudaContext& ctx,
                                           const SSElement& attr,
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

  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto output = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));
  auto grad_output = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[1]));
  auto input = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[2]));
  auto rois = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[3]));
  auto grad_rois = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);
  PARROTS_CALLDIOPI(diopiPrroiPoolCoorBackward(ch, output, grad_output, input, rois, grad_rois, pooled_height,
                                               pooled_width, spatial_scale));
}

PARROTS_EXTENSION_REGISTER(prroi_pool_forward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("spatial_scale")
    .input(2)
    .output(1)
    .apply(prroi_pool_forward_cuda_parrots_diopi)
    .done();

PARROTS_EXTENSION_REGISTER(prroi_pool_backward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("spatial_scale")
    .input(2)
    .output(1)
    .apply(prroi_pool_backward_cuda_parrots_diopi)
    .done();

PARROTS_EXTENSION_REGISTER(prroi_pool_coor_backward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("spatial_scale")
    .input(4)
    .output(1)
    .apply(prroi_pool_coor_backward_cuda_parrots_diopi)
    .done();
#endif
