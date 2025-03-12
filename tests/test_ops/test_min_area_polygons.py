# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.ops import min_area_polygons
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MUSA_AVAILABLE

np_pointsets = np.asarray([[
    1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0,
    2.0, 1.5, 1.5
],
                           [
                               1.0, 1.0, 8.0, 8.0, 1.0, 2.0, 2.0, 1.0, 1.0,
                               3.0, 3.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.5, 1.5
                           ]])

expected_polygons = np.asarray(
    [[3.0000, 1.0000, 1.0000, 1.0000, 1.0000, 3.0000, 3.0000, 3.0000],
     [8.0, 8.0, 2.3243, 0.0541, 0.0541, 1.6757, 5.7297, 9.6216]])


@pytest.mark.parametrize('device', [
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(
            not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
    pytest.param(
        'musa',
        marks=pytest.mark.skipif(
            not IS_MUSA_AVAILABLE, reason='requires MUSA support'))
])
def test_min_area_polygons(device):
    pointsets = torch.from_numpy(np_pointsets).to(device).float()

    assert np.allclose(
        min_area_polygons(pointsets).cpu().numpy(),
        expected_polygons,
        atol=1e-4)
