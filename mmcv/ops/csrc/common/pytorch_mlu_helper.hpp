#ifndef PYTORCH_MLU_HELPER
#define PYTORCH_MLU_HELPER

#ifdef MMCV_WITH_MLU
#include "aten.h"
#include "../pytorch/mlu/bang_internal.h"

#define NFU_ALIGN_SIZE 128

#define PAD_UP(x, y) (((x) / (y) + (int)((x) % (y) > 0)) * (y))

#define PAD_DOWN(x, y) (((x) / (y) ) * (y))

#endif

#endif  // PYTORCH_MLU_HELPER
