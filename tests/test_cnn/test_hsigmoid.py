# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.cnn.bricks import HSigmoid


def test_hsigmoid():
    # test assertion divisor can not be zero
    with pytest.raises(AssertionError):
        HSigmoid(divisor=0)

    # test with default parameters
    act = HSigmoid()
    input_shape = torch.Size([1, 3, 64, 64])
    input = torch.randn(input_shape)
    output = act(input)
    expected_output = torch.min(
        torch.max((input + 3) / 6, torch.zeros(input_shape)),
        torch.ones(input_shape))
    # test output shape
    assert output.shape == expected_output.shape
    # test output value
    assert torch.equal(output, expected_output)

    # test with designated parameters
    act = HSigmoid(1, 2, 0, 1)
    input_shape = torch.Size([1, 3, 64, 64])
    input = torch.randn(input_shape)
    output = act(input)
    expected_output = torch.min(
        torch.max((input + 1) / 2, torch.zeros(input_shape)),
        torch.ones(input_shape))
    # test output shape
    assert output.shape == expected_output.shape
    # test output value
    assert torch.equal(output, expected_output)
