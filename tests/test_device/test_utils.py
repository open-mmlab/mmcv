# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.device import get_device
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE, IS_MPS_AVAILABLE


def test_get_device():
    device = get_device()
    if IS_CUDA_AVAILABLE:
        assert device == 'cuda'
    elif IS_MLU_AVAILABLE:
        assert device == 'mlu'
    elif IS_MPS_AVAILABLE:
        assert device == 'mps'
    else:
        assert device == 'cpu'
