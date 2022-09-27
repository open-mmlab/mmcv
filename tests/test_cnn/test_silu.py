# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmcv.cnn.bricks import build_activation_layer


def test_silu():
    act = build_activation_layer(dict(type='SiLU'))
    input = torch.randn(1, 3, 64, 64)
    expected_output = input * torch.sigmoid(input)
    output = act(input)
    # test output shape
    assert output.shape == expected_output.shape
    # test output value
    assert torch.allclose(output, expected_output)

    # test inplace
    act = build_activation_layer(dict(type='SiLU', inplace=True))
    assert act.inplace
    input = torch.randn(1, 3, 64, 64)
    expected_output = input * torch.sigmoid(input)
    output = act(input)
    # test output shape
    assert output.shape == expected_output.shape
    # test output value
    assert torch.allclose(output, expected_output)
    assert torch.allclose(input, expected_output)
    assert input is output
