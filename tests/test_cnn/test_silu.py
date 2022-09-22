# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmcv.cnn.bricks import build_activation_layer


def test_silu():
    act = build_activation_layer(dict(type='SiLU'))
    input = torch.randn(1, 3, 64, 64)
    expected_output = F.silu(input)
    output = act(input)
    # test output shape
    assert output.shape == expected_output.shape
    # test output value
    assert torch.equal(output, expected_output)
