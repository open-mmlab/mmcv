# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.ops import points_in_polygons
from mmcv.utils import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE


@pytest.mark.parametrize('device', [
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(
            not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
    pytest.param(
        'npu',
        marks=pytest.mark.skipif(
            not IS_NPU_AVAILABLE, reason='requires NPU support'))
])
def test_points_in_polygons(device):
    points = np.array([[300., 300.], [400., 400.], [100., 100], [300, 250],
                       [100, 0]])
    polygons = np.array([[200., 200., 400., 400., 500., 200., 400., 100.],
                         [400., 400., 500., 500., 600., 300., 500., 200.],
                         [300., 300., 600., 700., 700., 700., 700., 100.]])
    expected_output = np.array([[0., 0., 0.], [0., 0., 1.], [0., 0., 0.],
                                [1., 0., 0.], [0., 0., 0.]]).astype(np.float32)
    points = torch.tensor(points, dtype=torch.float32, device=device)
    polygons = torch.tensor(polygons, dtype=torch.float32, device=device)
    assert np.allclose(
        points_in_polygons(points, polygons).cpu().numpy(), expected_output,
        1e-3)
