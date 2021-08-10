import pytest
import torch

from mmcv.cnn.bricks import ConvModule
from mmcv.cnn.utils import revert_sync_batchnorm


def test_revert_sync_batchnorm():
    conv = ConvModule(3, 8, 2, norm_cfg=dict(type='SyncBN'))
    x = torch.randn(1, 3, 10, 10)
    with pytest.raises(ValueError):
        y = conv(x)
    conv = revert_sync_batchnorm(conv)
    y = conv(x)
    assert y.shape == (1, 8, 9, 9)
