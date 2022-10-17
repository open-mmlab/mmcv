# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import patch

import torch.nn as nn

from mmcv.device.mps import MPSDataParallel
from mmcv.parallel import is_module_wrapper
from mmcv.utils import IS_MPS_AVAILABLE


def mock(*args, **kwargs):
    pass


@patch('torch.distributed._broadcast_coalesced', mock)
@patch('torch.distributed.broadcast', mock)
@patch('torch.nn.parallel.DistributedDataParallel._ddp_init_helper', mock)
def test_is_module_wrapper():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 2, 1)

        def forward(self, x):
            return self.conv(x)

    model = Model()
    assert not is_module_wrapper(model)

    if IS_MPS_AVAILABLE:
        mpsdp = MPSDataParallel(model)
        assert is_module_wrapper(mpsdp)
