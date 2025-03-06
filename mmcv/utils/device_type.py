# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.device import is_cuda_available

IS_CUDA_AVAILABLE = is_cuda_available()
IS_MLU_AVAILABLE = False
IS_NPU_AVAILABLE = False
IS_MPS_AVAILABLE = False
IS_MUSA_AVAILABLE = False
