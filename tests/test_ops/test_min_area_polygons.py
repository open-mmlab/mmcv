import numpy as np
import pytest
import torch

from mmcv.ops import min_area_polygons

np_pointsets = np.asarray([[
    1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0,
    2.0, 1.5, 1.5, 9.5, 9.5
],
                           [
                               1.5, 1.5, 2.5, 2.5, 1.5, 2.5, 2.5, 1.5, 1.5,
                               3.5, 3.5, 1.5, 2.5, 3.5, 3.5, 2.5, 2.0, 2.0,
                               5.5, 5.5
                           ]])

expected_polygons = np.asarray(
    [[3.0000, 1.0000, 1.0000, 1.0000, 1.0000, 3.0000, 3.0000, 3.0000],
     [9.5000, 9.5000, 2.7800, 0.5400, 0.5400, 2.2200, 7.2600, 11.1800]])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_min_area_polygons():
    pointsets = torch.from_numpy(np_pointsets).cuda().float()

    assert np.allclose(
        min_area_polygons(pointsets).cpu().numpy(),
        expected_polygons,
        atol=1e-4)
