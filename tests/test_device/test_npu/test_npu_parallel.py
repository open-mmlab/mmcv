# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock, patch

import torch.nn as nn

from mmcv.device.npu import NPUDataParallel, NPUDistributedDataParallel
from mmcv.parallel import is_module_wrapper
from mmcv.utils import IS_NPU_AVAILABLE


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

    if IS_NPU_AVAILABLE:
        npudp = NPUDataParallel(model)
        assert is_module_wrapper(npudp)

        npuddp = NPUDistributedDataParallel(model, process_group=MagicMock())
        assert is_module_wrapper(npuddp)
