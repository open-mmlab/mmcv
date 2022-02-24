# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.ops import furthest_point_sample, furthest_point_sample_with_dist


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_fps():
    xyz = torch.tensor([[[-0.2748, 1.0020, -1.1674], [0.1015, 1.3952, -1.2681],
                         [-0.8070, 2.4137,
                          -0.5845], [-1.0001, 2.1982, -0.5859],
                         [0.3841, 1.8983, -0.7431]],
                        [[-1.0696, 3.0758,
                          -0.1899], [-0.2559, 3.5521, -0.1402],
                         [0.8164, 4.0081, -0.1839], [-1.1000, 3.0213, -0.8205],
                         [-0.0518, 3.7251, -0.3950]]]).cuda()

    idx = furthest_point_sample(xyz, 3)
    expected_idx = torch.tensor([[0, 2, 4], [0, 2, 1]]).cuda()
    assert torch.all(idx == expected_idx)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_fps_with_dist():
    xyz = torch.tensor([[[-0.2748, 1.0020, -1.1674], [0.1015, 1.3952, -1.2681],
                         [-0.8070, 2.4137,
                          -0.5845], [-1.0001, 2.1982, -0.5859],
                         [0.3841, 1.8983, -0.7431]],
                        [[-1.0696, 3.0758,
                          -0.1899], [-0.2559, 3.5521, -0.1402],
                         [0.8164, 4.0081, -0.1839], [-1.1000, 3.0213, -0.8205],
                         [-0.0518, 3.7251, -0.3950]]]).cuda()

    expected_idx = torch.tensor([[0, 2, 4], [0, 2, 1]]).cuda()
    xyz_square_dist = ((xyz.unsqueeze(dim=1) -
                        xyz.unsqueeze(dim=2))**2).sum(-1)
    idx = furthest_point_sample_with_dist(xyz_square_dist, 3)
    assert torch.all(idx == expected_idx)

    import numpy as np
    fps_idx = np.load('tests/data/for_3d_ops/fps_idx.npy')
    features_for_fps_distance = np.load(
        'tests/data/for_3d_ops/features_for_fps_distance.npy')
    expected_idx = torch.from_numpy(fps_idx).cuda()
    features_for_fps_distance = torch.from_numpy(
        features_for_fps_distance).cuda()

    idx = furthest_point_sample_with_dist(features_for_fps_distance, 16)
    assert torch.all(idx == expected_idx)
