import numpy as np
import torch

from mmcv.cnn.bricks import HSigmoid


def test_hsigmoid():
    act = HSigmoid()
    input_shape = torch.Size([1, 3, 64, 64])
    input = torch.randn(input_shape)
    output = act(input)
    expected_output = torch.min(
        torch.max((input + 1) / 2, torch.zeros(input_shape)),
        torch.ones(input_shape))
    # test output shape
    assert output.shape == expected_output.shape
    # test output value
    assert np.equal(output.cpu().data.numpy(),
                    expected_output.cpu().data.numpy()).all()
