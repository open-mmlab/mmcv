# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE, IS_MPS_AVAILABLE


def get_device() -> str:
    """Returns the currently existing device type.

    Returns:
        str: cuda | mlu | mps | cpu.
    """
    if IS_CUDA_AVAILABLE:
        return 'cuda'
    elif IS_MLU_AVAILABLE:
        return 'mlu'
    elif IS_MPS_AVAILABLE:
        return 'mps'
    else:
        return 'cpu'
