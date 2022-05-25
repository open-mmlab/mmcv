# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.ops import Voxelization


def _get_voxel_points_indices(points, coors, voxel):
    result_form = np.equal(coors, voxel)
    return result_form[:, 0] & result_form[:, 1] & result_form[:, 2]


@pytest.mark.parametrize('device_type', [
    'cpu',
    pytest.param(
        'cuda:0',
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason='requires CUDA support'))
])
def test_voxelization(device_type):
    voxel_size = [0.5, 0.5, 0.5]
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]

    voxel_dict = np.load(
        'tests/data/for_3d_ops/test_voxel.npy', allow_pickle=True).item()
    expected_coors = voxel_dict['coors']
    expected_voxels = voxel_dict['voxels']
    expected_num_points_per_voxel = voxel_dict['num_points_per_voxel']
    points = voxel_dict['points']

    points = torch.tensor(points)
    max_num_points = -1
    dynamic_voxelization = Voxelization(voxel_size, point_cloud_range,
                                        max_num_points)
    max_num_points = 1000
    hard_voxelization = Voxelization(voxel_size, point_cloud_range,
                                     max_num_points)

    device = torch.device(device_type)

    # test hard_voxelization on cpu/gpu
    points = points.contiguous().to(device)
    coors, voxels, num_points_per_voxel = hard_voxelization.forward(points)
    coors = coors.cpu().detach().numpy()
    voxels = voxels.cpu().detach().numpy()
    num_points_per_voxel = num_points_per_voxel.cpu().detach().numpy()
    assert np.all(coors == expected_coors)
    assert np.all(voxels == expected_voxels)
    assert np.all(num_points_per_voxel == expected_num_points_per_voxel)

    # test dynamic_voxelization on cpu/gpu
    coors = dynamic_voxelization.forward(points)
    coors = coors.cpu().detach().numpy()
    points = points.cpu().detach().numpy()
    for i in range(expected_voxels.shape[0]):
        indices = _get_voxel_points_indices(points, coors, expected_voxels[i])
        num_points_current_voxel = points[indices].shape[0]
        assert num_points_current_voxel > 0
        assert np.all(
            points[indices] == expected_coors[i][:num_points_current_voxel])
        assert num_points_current_voxel == expected_num_points_per_voxel[i]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_voxelization_nondeterministic():
    voxel_size = [0.5, 0.5, 0.5]
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]

    voxel_dict = np.load(
        'tests/data/for_3d_ops/test_voxel.npy', allow_pickle=True).item()
    points = voxel_dict['points']

    points = torch.tensor(points)
    max_num_points = -1
    dynamic_voxelization = Voxelization(voxel_size, point_cloud_range,
                                        max_num_points)

    max_num_points = 10
    max_voxels = 50
    hard_voxelization = Voxelization(
        voxel_size,
        point_cloud_range,
        max_num_points,
        max_voxels,
        deterministic=False)

    # test hard_voxelization (non-deterministic version) on gpu
    points = torch.tensor(points).contiguous().to(device='cuda:0')
    voxels, coors, num_points_per_voxel = hard_voxelization.forward(points)
    coors = coors.cpu().detach().numpy().tolist()
    voxels = voxels.cpu().detach().numpy().tolist()
    num_points_per_voxel = num_points_per_voxel.cpu().detach().numpy().tolist()

    coors_all = dynamic_voxelization.forward(points)
    coors_all = coors_all.cpu().detach().numpy().tolist()

    coors_set = {tuple(c) for c in coors}
    coors_all_set = {tuple(c) for c in coors_all}

    assert len(coors_set) == len(coors)
    assert len(coors_set - coors_all_set) == 0

    points = points.cpu().detach().numpy().tolist()

    coors_points_dict = {}
    for c, ps in zip(coors_all, points):
        if tuple(c) not in coors_points_dict:
            coors_points_dict[tuple(c)] = set()
        coors_points_dict[tuple(c)].add(tuple(ps))

    for c, ps, n in zip(coors, voxels, num_points_per_voxel):
        ideal_voxel_points_set = coors_points_dict[tuple(c)]
        voxel_points_set = {tuple(p) for p in ps[:n]}
        assert len(voxel_points_set) == n
        if n < max_num_points:
            assert voxel_points_set == ideal_voxel_points_set
            for p in ps[n:]:
                assert max(p) == min(p) == 0
        else:
            assert len(voxel_points_set - ideal_voxel_points_set) == 0

    # test hard_voxelization (non-deterministic version) on gpu
    # with all input point in range
    points = torch.tensor(points).contiguous().to(device='cuda:0')[:max_voxels]
    coors_all = dynamic_voxelization.forward(points)
    valid_mask = coors_all.ge(0).all(-1)
    points = points[valid_mask]
    coors_all = coors_all[valid_mask]
    coors_all = coors_all.cpu().detach().numpy().tolist()

    voxels, coors, num_points_per_voxel = hard_voxelization.forward(points)
    coors = coors.cpu().detach().numpy().tolist()

    coors_set = {tuple(c) for c in coors}
    coors_all_set = {tuple(c) for c in coors_all}

    assert len(coors_set) == len(coors) == len(coors_all_set)
