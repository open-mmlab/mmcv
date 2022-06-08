# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.ops import chamfer_distance


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_chamfer_distance():
    pointset1 = torch.tensor(
        [[[1.3, 9.39], [2.3, 9.39], [2.3, 10.39], [1.3, 10.39]],
         [[1.0, 9.39], [3.0, 9.39], [3.0, 10.39], [1.0, 10.39]],
         [[1.6, 9.99], [2.3, 9.99], [2.3, 10.39], [1.6, 10.39]]],
        device='cuda',
        requires_grad=True)

    pointset2 = torch.tensor(
        [[[1.0, 9.39], [3.0, 9.39], [3.0, 10.39], [1.0, 10.39]],
         [[1.3, 9.39], [2.3, 9.39], [2.3, 10.39], [1.3, 10.39]],
         [[1.0, 9.39], [3.0, 9.39], [3.0, 10.39], [1.0, 10.39]]],
        device='cuda',
        requires_grad=True)

    expected_dist1 = torch.tensor(
        [[0.0900, 0.4900, 0.4900, 0.0900], [0.0900, 0.4900, 0.4900, 0.0900],
         [0.5200, 0.6500, 0.4900, 0.3600]],
        device='cuda')
    expected_dist2 = torch.tensor(
        [[0.0900, 0.4900, 0.4900, 0.0900], [0.0900, 0.4900, 0.4900, 0.0900],
         [0.7200, 0.8500, 0.4900, 0.3600]],
        device='cuda')

    expected_pointset1_grad = torch.tensor(
        [[[0.6000, 0.0000], [-1.4000, 0.0000], [-1.4000, 0.0000],
          [0.6000, 0.0000]],
         [[-0.6000, 0.0000], [1.4000, 0.0000], [1.4000, 0.0000],
          [-0.6000, 0.0000]],
         [[1.2000, -0.8000], [-1.4000, -0.8000], [-1.4000, 0.0000],
          [1.2000, 0.0000]]],
        device='cuda')

    expected_pointset2_grad = torch.tensor(
        [[[-0.6000, 0.0000], [1.4000, 0.0000], [1.4000, 0.0000],
          [-0.6000, 0.0000]],
         [[0.6000, 0.0000], [-1.4000, 0.0000], [-1.4000, 0.0000],
          [0.6000, 0.0000]],
         [[0.0000, 0.0000], [0.0000, 0.0000], [2.8000, 0.8000],
          [-2.4000, 0.8000]]],
        device='cuda')

    dist1, dist2, idx1, idx2 = chamfer_distance(pointset1, pointset2)
    dist1.backward(torch.ones_like(dist1))
    assert torch.allclose(dist1, expected_dist1, 1e-2)
    assert torch.allclose(dist2, expected_dist2, 1e-2)
    assert torch.allclose(pointset1.grad.data, expected_pointset1_grad, 1e-2)
    assert torch.allclose(pointset2.grad.data, expected_pointset2_grad, 1e-2)
