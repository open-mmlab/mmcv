// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "min_area_polygons_pytorch.h"
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <parrots/diopi.hpp>

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void min_area_polygons_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                    const OperatorBase::in_list_t& ins,
                                    OperatorBase::out_list_t& outs) {
  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto pointsets = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));

  auto polygons = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);
  PARROTS_CALLDIOPI(diopiMinAreaPolygons(ch, pointsets, polygons));
}

PARROTS_EXTENSION_REGISTER(min_area_polygons)
    .input(1)
    .output(1)
    .apply(min_area_polygons_cuda_parrots)
    .done();

#endif
