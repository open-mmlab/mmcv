# Copyright (c) Open-MMLab. All rights reserved.
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import mmcv

try:
    import torch
except ImportError:
    torch = None


@pytest.mark.skipif(torch is None, reason='requires torch library')
def test_tensor2imgs():

    # test tensor obj
    with pytest.raises(AssertionError):
        tensor = np.random.rand(2, 3, 3)
        mmcv.tensor2imgs(tensor)

    # test tensor ndim
    with pytest.raises(AssertionError):
        tensor = torch.randn(2, 3, 3)
        mmcv.tensor2imgs(tensor)

    # test mean length
    with pytest.raises(AssertionError):
        tensor = torch.randn(2, 3, 5, 5)
        mmcv.tensor2imgs(tensor, mean=(1, ))

    # test std length
    with pytest.raises(AssertionError):
        tensor = torch.randn(2, 3, 5, 5)
        mmcv.tensor2imgs(tensor, std=(1, ))

    # test rgb=True
    tensor = torch.randn(2, 3, 5, 5)
    gts = [
        t.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        for t in tensor.flip(1)
    ]
    outputs = mmcv.tensor2imgs(tensor, to_rgb=True)
    for gt, output in zip(gts, outputs):
        assert_array_equal(gt, output)

    # test rgb=False
    tensor = torch.randn(2, 3, 5, 5)
    gts = [t.cpu().numpy().transpose(1, 2, 0).astype(np.uint8) for t in tensor]
    outputs = mmcv.tensor2imgs(tensor, to_rgb=False)
    for gt, output in zip(gts, outputs):
        assert_array_equal(gt, output)
