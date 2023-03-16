// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "min_area_polygons_pytorch.h"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>
#include <parrots/diopi.hpp>
#endif

using namespace parrots;

#ifdef MMCV_WITH_CUDA
#ifdef MMCV_WITH_DIOPI
void min_area_polygons_cuda_parrots_diopi(CudaContext& ctx, const SSElement& attr,
                                    const OperatorBase::in_list_t& ins,
                                    OperatorBase::out_list_t& outs) {
  diopiContext dctx(ctx);
  diopiContextHandle_t ch = &dctx;
  auto pointsets = reinterpret_cast<diopiTensorHandle_t>(const_cast<DArray*>(&ins[0]));

  auto polygons = reinterpret_cast<diopiTensorHandle_t>(&outs[0]);
  PARROTS_CALLDIOPI(diopiMinAreaPolygons(ch, pointsets, polygons));
}
#else
void min_area_polygons_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                    const OperatorBase::in_list_t& ins,
                                    OperatorBase::out_list_t& outs) {
  auto pointsets = buildATensor(ctx, ins[0]);

  auto polygons = buildATensor(ctx, outs[0]);
  min_area_polygons(pointsets, polygons);
}
#endif

PARROTS_EXTENSION_REGISTER(min_area_polygons)
    .input(1)
    .output(1)
#ifdef MMCV_WITH_DIOPI
    .apply(min_area_polygons_cuda_parrots_diopi)
#else
    .apply(min_area_polygons_cuda_parrots)
#endif
    .done();

#endif
