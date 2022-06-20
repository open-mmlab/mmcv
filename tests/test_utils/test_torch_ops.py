# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.utils import torch_meshgrid


def test_torch_meshgrid():
    # torch_meshgrid should not throw warning
    with pytest.warns(None) as record:
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        grid_x, grid_y = torch_meshgrid(x, y)

    assert len(record) == 0
