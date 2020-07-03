import torch

from mmcv.cnn import ConvModule, fuse_module


def test_fuse_module():
    inputs = torch.rand((1, 3, 5, 5))
    module = ConvModule(3, 5, 3, norm_cfg=dict(type='BN'))
    fused_module = fuse_module(module)
    assert torch.equal(module(inputs), fused_module(inputs))
