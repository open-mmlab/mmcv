from unittest.mock import MagicMock, patch

import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from mmcv.parallel import (MMDataParallel, MMDistributedDataParallel,
                           is_parallel_module)
from mmcv.parallel.distributed_deprecated import \
    MMDistributedDataParallel as DeprecatedMMDDP


@patch('torch.distributed._broadcast_coalesced', MagicMock)
@patch('torch.distributed.broadcast', MagicMock)
@patch('torch.nn.parallel.DistributedDataParallel._ddp_init_helper', MagicMock)
def test_is_parallel_module():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 2, 1)

        def forward(self, x):
            return self.conv(x)

    model = Model()
    assert not is_parallel_module(model)

    dp = DataParallel(model)
    assert is_parallel_module(dp)

    mmdp = MMDataParallel(model)
    assert is_parallel_module(mmdp)

    ddp = DistributedDataParallel(model, process_group=MagicMock())
    assert is_parallel_module(ddp)

    mmddp = MMDistributedDataParallel(model, process_group=MagicMock())
    assert is_parallel_module(mmddp)

    deprecated_mmddp = DeprecatedMMDDP(model)
    assert is_parallel_module(deprecated_mmddp)
