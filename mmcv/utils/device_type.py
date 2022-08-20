# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.device import is_mlu_available, is_mps_available, is_cuda_available


IS_MLU_AVAILABLE = is_mlu_available()
IS_MPS_AVAILABLE = is_mps_available()
IS_CUDA_AVAILABLE = is_cuda_available()
