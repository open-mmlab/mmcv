# Copyright (c) OpenMMLab. All rights reserved.
from .device_type import (IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE,
                          IS_MPS_AVAILABLE, IS_NPU_AVAILABLE)
from .env import collect_env

__all__ = [
    'IS_MLU_AVAILABLE', 'IS_MPS_AVAILABLE', 'IS_CUDA_AVAILABLE',
    'IS_NPU_AVAILABLE', 'collect_env'
]
