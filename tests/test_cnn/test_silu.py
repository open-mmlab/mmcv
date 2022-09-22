# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmcv.cnn.bricks import build_activation_layer


def test_silu():
    act = build_activation_layer(dict(type='SiLU'))
    input = torch.randn(1, 3, 64, 64)
    output = act(input)
    expected_output = input * torch.sigmoid(input)
    # test output shape
    assert output.shape == expected_output.shape
    # test output value
    print(output)
    print(expected_output)
    assert torch.allclose(output, expected_output)
