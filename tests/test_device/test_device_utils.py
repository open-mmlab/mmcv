# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.device import get_device
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE, IS_MPS_AVAILABLE


def test_get_device():
    current_device = get_device()
    if IS_CUDA_AVAILABLE:
        assert current_device == 'cuda'
    elif IS_MLU_AVAILABLE:
        assert current_device == 'mlu'
    elif IS_MPS_AVAILABLE:
        assert current_device == 'mps'
    else:
        assert current_device == 'cpu'
