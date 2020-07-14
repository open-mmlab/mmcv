import torch
import torch.nn as nn

from mmcv.cnn import ConvModule, fuse_conv_bn


def test_fuse_conv_bn():
    inputs = torch.rand((1, 3, 5, 5))
    modules = nn.ModuleList()
    modules.append(nn.BatchNorm2d(3))
    modules.append(ConvModule(3, 5, 3, norm_cfg=dict(type='BN')))
    modules.append(ConvModule(5, 5, 3, norm_cfg=dict(type='BN')))
    modules = nn.Sequential(*modules)
    fused_modules = fuse_conv_bn(modules)
    assert torch.equal(modules(inputs), fused_modules(inputs))
