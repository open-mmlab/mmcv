# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for the optimized deterministic hard_voxelize path (hash + sort).

The optimized path must keep byte-identical output to the legacy
implementation, so besides the official fixed-data comparison we also verify
the output against dynamic voxelization (which is order-agnostic ground
truth) on random inputs, covering content, per-voxel point order and
first-occurrence voxel numbering.
"""
import numpy as np
import pytest
import torch

from mmcv.ops import Voxelization
from mmcv.utils import IS_CUDA_AVAILABLE


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason='requires CUDA support')
def test_voxelization_v2_official_expected():
    voxel_size = [0.5, 0.5, 0.5]
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]

    voxel_dict = np.load(
        'tests/data/for_3d_ops/test_voxel.npy', allow_pickle=True).item()
    expected_coors = voxel_dict['coors']
    expected_voxels = voxel_dict['voxels']
    expected_num_points_per_voxel = voxel_dict['num_points_per_voxel']
    points = torch.tensor(voxel_dict['points']).contiguous().to('cuda:0')

    max_num_points = 1000
    hard_voxelization = Voxelization(voxel_size, point_cloud_range,
                                     max_num_points)
    coors, voxels, num_points_per_voxel = hard_voxelization.forward(points)

    coors = coors.cpu().detach().numpy()
    voxels = voxels.cpu().detach().numpy()
    num_points_per_voxel = num_points_per_voxel.cpu().detach().numpy()
    assert np.all(coors == expected_coors)
    assert np.all(voxels == expected_voxels)
    assert np.all(num_points_per_voxel == expected_num_points_per_voxel)


def _group_by_coors(points_np, coors_np):
    """Group original points by their dynamic-voxelization coors.

    Returns dict: key=(cz, cy, cx) -> list of original point rows in the
    original order.  Out-of-range rows (coors[0] == -1) are skipped.
    """
    groups = {}
    for row, c in zip(points_np, coors_np):
        if c[0] < 0:
            continue
        groups.setdefault(tuple(int(v) for v in c), []).append(tuple(row))
    return groups


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason='requires CUDA support')
def test_voxelization_v2_random_equiv_dynamic():
    voxel_size = [0.5, 0.5, 0.5]
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    max_num_points = 10
    max_voxels = 200

    np.random.seed(0)
    # mix of in-range and out-of-range points on each axis
    n = 500
    points = np.zeros((n, 4), dtype=np.float32)
    points[:, 0] = np.random.uniform(-5, 75, n)  # partly < 0 or > 70.4
    points[:, 1] = np.random.uniform(-45, 45, n)  # partly out of [-40, 40]
    points[:, 2] = np.random.uniform(-4, 2, n)  # partly out of [-3, 1]
    points[:, 3] = np.random.rand(n)
    points = torch.tensor(points).contiguous().to('cuda:0')

    dynamic_voxelization = Voxelization(voxel_size, point_cloud_range, -1)
    coors_all = dynamic_voxelization.forward(points).cpu().detach().numpy()

    hard_voxelization = Voxelization(voxel_size, point_cloud_range,
                                     max_num_points, max_voxels)
    voxels, coors, num_points_per_voxel = hard_voxelization.forward(points)
    coors = coors.cpu().detach().numpy()
    voxels = voxels.cpu().detach().numpy()
    num_points_per_voxel = num_points_per_voxel.cpu().detach().numpy()

    groups = _group_by_coors(points.cpu().detach().numpy(), coors_all)

    # 1. all output voxel coors are unique and appear in the dynamic groups
    assert len(coors) == len(set(map(tuple, coors)))
    assert set(map(tuple, coors)) <= set(groups.keys())
    assert len(coors) <= max_voxels

    # 2. voxel numbering follows first-occurrence order of each (cz, cy, cx)
    first_pos = {}
    for i, c in enumerate(coors_all):
        if c[0] >= 0:
            key = tuple(int(v) for v in c)
            first_pos.setdefault(key, i)
    expected_coors_order = sorted(coors, key=lambda c: first_pos[tuple(c)])
    assert np.all(coors == expected_coors_order)

    # 3. per-voxel content: first min(n, max_points) points in original order
    for voxel_idx, (c, n_p) in enumerate(zip(coors, num_points_per_voxel)):
        ideal = groups[tuple(c)]
        expected_n = min(len(ideal), max_num_points)
        assert n_p == expected_n
        got = [tuple(row) for row in voxels[voxel_idx][:n_p]]
        assert got == ideal[:expected_n]
        # tail stays zero-padded
        assert np.all(voxels[voxel_idx][n_p:] == 0)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason='requires CUDA support')
def test_voxelization_v2_max_voxels_cap():
    voxel_size = [0.5, 0.5, 0.5]
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    max_num_points = 35
    max_voxels = 5

    np.random.seed(1)
    points = np.random.uniform(-40, 40, (2000, 3)).astype(np.float32)
    points = torch.tensor(points).contiguous().to('cuda:0')

    hard_voxelization = Voxelization(voxel_size, point_cloud_range,
                                     max_num_points, max_voxels)
    voxels, coors, num_points_per_voxel = hard_voxelization.forward(points)
    assert coors.size(0) <= max_voxels
    coors = coors.cpu().detach().numpy()
    assert len(set(map(tuple, coors))) == coors.shape[0]


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason='requires CUDA support')
def test_voxelization_v2_empty_and_single():
    voxel_size = [0.5, 0.5, 0.5]
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    hard_voxelization = Voxelization(voxel_size, point_cloud_range, 35, 200)

    # empty input
    points = torch.empty((0, 3), dtype=torch.float32).to('cuda:0')
    voxels, coors, num_points_per_voxel = hard_voxelization.forward(points)
    assert coors.numel() == 0 and voxels.numel() == 0

    # single in-range point
    points = torch.tensor([[1.0, 1.0, 0.5, 0.5]],
                          dtype=torch.float32).to('cuda:0')
    voxels, coors, num_points_per_voxel = hard_voxelization.forward(points)
    assert coors.size(0) == 1
    assert int(num_points_per_voxel[0]) == 1
    assert np.all(voxels[0][0].cpu().numpy() == [1.0, 1.0, 0.5, 0.5])


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason='requires CUDA support')
def test_voxelization_v2_fallback_dtype_smoke():
    # Non-float32 input never reaches the optimized path: the launcher hands it
    # to the legacy code, whose CUDA kernels are float32-only.  What is pinned
    # down here is that the optimization leaves that behavior untouched.
    voxel_size = [0.5, 0.5, 0.5]
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    hard_voxelization = Voxelization(voxel_size, point_cloud_range, 35, 200)

    points = torch.tensor([[1.0, 1.0, 0.5, 0.5], [1.0, 1.0, 0.5, 0.6]],
                          dtype=torch.float64).to('cuda:0')
    with pytest.raises(RuntimeError, match='Float'):
        hard_voxelization.forward(points)

    with pytest.raises(RuntimeError):
        hard_voxelization.forward(points.to(torch.float16))
