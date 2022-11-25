# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.cnn.bricks import LayerScale, Scale


def test_scale():
    # test default scale
    scale = Scale()
    assert scale.scale.data == 1.
    assert scale.scale.dtype == torch.float
    x = torch.rand(1, 3, 64, 64)
    output = scale(x)
    assert output.shape == (1, 3, 64, 64)

    # test given scale
    scale = Scale(10.)
    assert scale.scale.data == 10.
    assert scale.scale.dtype == torch.float
    x = torch.rand(1, 3, 64, 64)
    output = scale(x)
    assert output.shape == (1, 3, 64, 64)


def test_layer_scale():
    with pytest.raises(AssertionError):
        cfg = dict(
            dim=10,
            data_format='BNC',
        )
        LayerScale(**cfg)

    # test init
    cfg = dict(dim=10)
    ls = LayerScale(**cfg)
    assert torch.equal(ls.weight, torch.ones(10, requires_grad=True) * 1e-5)

    # test forward
    # test channels_last
    cfg = dict(dim=256, inplace=False, data_format='channels_last')
    ls_channels_last = LayerScale(**cfg)
    x = torch.randn((4, 49, 256))
    out = ls_channels_last(x)
    assert tuple(out.size()) == (4, 49, 256)
    assert torch.equal(x * 1e-5, out)

    # test channels_first
    cfg = dict(dim=256, inplace=False, data_format='channels_first')
    ls_channels_first = LayerScale(**cfg)
    x = torch.randn((4, 256, 7, 7))
    out = ls_channels_first(x)
    assert tuple(out.size()) == (4, 256, 7, 7)
    assert torch.equal(x * 1e-5, out)

    # test inplace True
    cfg = dict(dim=256, inplace=True, data_format='channels_first')
    ls_channels_first = LayerScale(**cfg)
    x = torch.randn((4, 256, 7, 7))
    out = ls_channels_first(x)
    assert tuple(out.size()) == (4, 256, 7, 7)
    assert x is out
