# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.ops import ChamferDistance


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_chamfer_distance():
    point_set_1 = torch.tensor(
        [[[1.3, 9.39],
          [2.3, 9.39],
          [2.3, 10.39],
          [1.3, 10.39]],
         [[1.0, 9.39],
          [3.0, 9.39],
          [3.0, 10.39],
          [1.0, 10.39]],
         [[1.6, 9.99],
          [2.3, 9.99],
          [2.3, 10.39],
          [1.6, 10.39]]],
        device='cuda',
        requires_grad=True)

    point_set_2 = torch.tensor(
        [[[1.0, 9.39],
          [3.0, 9.39],
          [3.0, 10.39],
          [1.0, 10.39]],
         [[1.3, 9.39],
          [2.3, 9.39],
          [2.3, 10.39],
          [1.3, 10.39]],
         [[1.0, 9.39],
          [3.0, 9.39],
          [3.0, 10.39],
          [1.0, 10.39]]],
        device='cuda',
        requires_grad=True)

    expected_dist1 = torch.tensor([[0.5800, 0.4900, 0.1800, 0.4900],
                                   [0.5800, 0.5200, 1.1600, 0.3600],
                                   [0.0000, 0.0000, 0.0000, 0.0000]],
                                  device='cuda')
    expected_dist2 = torch.tensor([[0.5800, 0.4900, 0.1800, 0.4900],
                                   [0.5800, 0.7200, 1.3400, 0.3600],
                                   [0.0000, 0.0000, 0.0000, 0.0000]],
                                  device='cuda')

    chamfer = ChamferDistance()
    dist1, dist2, _, _ = chamfer(point_set_1, point_set_2)
    assert torch.allclose(dist1, expected_dist1, 1e-2)
    assert torch.allclose(dist2, expected_dist2, 1e-2)
