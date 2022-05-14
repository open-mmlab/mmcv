# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmcv.cnn.bricks import Scale


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
