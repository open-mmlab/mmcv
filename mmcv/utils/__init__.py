# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils.device_type import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE, IS_MPS_AVAILABLE, IS_MUSA_AVAILABLE, IS_NPU_AVAILABLE
from mmcv.utils.env import collect_env
from mmcv.utils.parrots_jit import jit, skip_no_elena
# ext_loader has been removed as we're using PyTorch-only implementations

__all__ = [
                          'IS_CUDA_AVAILABLE',
                          'IS_MLU_AVAILABLE',
                          'IS_MPS_AVAILABLE',
                          'IS_MUSA_AVAILABLE',
                          'IS_NPU_AVAILABLE',
                          'collect_env',
                          'jit',
                          'skip_no_elena'
]
