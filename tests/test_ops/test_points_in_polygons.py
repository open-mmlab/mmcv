# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.ops import points_in_polygons


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_points_in_polygons():
    points = np.array([[300., 300.], [400., 400.], [100., 100], [300, 250],
                       [100, 0]])
    polygons = np.array([[200., 200., 400., 400., 500., 200., 400., 100.],
                         [400., 400., 500., 500., 600., 300., 500., 200.],
                         [300., 300., 600., 700., 700., 700., 700., 100.]])
    expected_output = np.array([[0., 0., 0.], [0., 0., 1.], [0., 0., 0.],
                                [1., 0., 0.], [0., 0., 0.]])
    points = torch.from_numpy(points).cuda().float()
    polygons = torch.from_numpy(polygons).cuda().float()
    expected_output = torch.from_numpy(expected_output).cuda().float()
    assert torch.allclose(
        points_in_polygons(points, polygons), expected_output, 1e-3)
