# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn.functional as F

from mmcv.cnn.bricks import build_activation_layer
from mmcv.utils import digit_version


@pytest.mark.skipif(
    digit_version(torch.__version__) < digit_version('1.7.0'),
    reason='torch.nn.SiLU is not available before 1.7.0')
def test_silu():
    act = build_activation_layer(dict(type='SiLU'))
    input = torch.randn(1, 3, 64, 64)
    expected_output = F.silu(input)
    output = act(input)
    # test output shape
    assert output.shape == expected_output.shape
    # test output value
    assert torch.equal(output, expected_output)
