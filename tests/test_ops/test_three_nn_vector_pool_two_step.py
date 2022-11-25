# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.ops import three_nn_vector_pool_by_two_step
from mmcv.utils import IS_CUDA_AVAILABLE


def get_dense_voxels_by_center(point_centers, max_neighbour_distance,
                               num_voxels):
    """
    Args:
        point_centers: (N, 3)
        max_neighbour_distance: float
        num_voxels: [num_x, num_y, num_z]

    Returns:
        voxel_centers: (N, total_voxels, 3)
    """
    R = max_neighbour_distance
    device = point_centers.device
    x_grids = torch.arange(
        -R + R / num_voxels[0],
        R - R / num_voxels[0] + 1e-5,
        2 * R / num_voxels[0],
        device=device)
    y_grids = torch.arange(
        -R + R / num_voxels[1],
        R - R / num_voxels[1] + 1e-5,
        2 * R / num_voxels[1],
        device=device)
    z_grids = torch.arange(
        -R + R / num_voxels[2],
        R - R / num_voxels[2] + 1e-5,
        2 * R / num_voxels[2],
        device=device)
    x_offset, y_offset, z_offset = torch.meshgrid(
        x_grids, y_grids, z_grids)  # shape: [num_x, num_y, num_z]
    xyz_offset = torch.cat(
        (x_offset.contiguous().view(-1, 1), y_offset.contiguous().view(
            -1, 1), z_offset.contiguous().view(-1, 1)),
        dim=-1)
    voxel_centers = point_centers[:, None, :] + xyz_offset[None, :, :]
    return voxel_centers


@pytest.mark.parametrize('device', [
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(
            not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
])
def test_three_nn_vector_pool(device):
    xyz = torch.tensor(
        [[0.7911, 4.1821, 18.1309], [9.8552, 19.9272, 7.4532],
         [17.0715, 9.8851, 5.8078], [4.3750, 1.1232, 18.0702],
         [14.0227, 9.5781, 15.7914], [3.0038, 8.7471, 12.6253],
         [17.1353, 13.0427, 13.4723], [1.4284, 12.0409, 16.0280],
         [10.5802, 11.9821, 10.6400], [11.2924, 16.3918, 16.3261],
         [8.6749, 4.3318, 19.6607], [6.7047, 10.6616, 16.7599],
         [15.1153, 1.8694, 16.1620], [4.5372, 2.2882, 12.4915],
         [12.0136, 0.5850, 4.2164], [15.2224, 13.8230, 19.8346],
         [16.7076, 12.8573, 5.8789], [17.8641, 18.0247, 0.7161],
         [12.7604, 10.6771, 19.1813], [10.3219, 10.4839, 14.7624]],
        device=device)
    new_xyz = torch.tensor(
        [[0.1411, 15.6141, 9.3022], [15.6595, 0.9505, 19.3470],
         [8.0824, 10.3586, 17.3501], [7.3926, 9.9670, 6.6586],
         [13.8781, 8.9048, 5.8824], [11.1121, 0.0274, 9.4883],
         [0.4287, 1.5586, 6.9646], [2.7858, 1.8852, 15.0609],
         [6.0411, 2.8716, 18.9102], [9.1480, 10.8151, 17.0509],
         [5.1243, 8.9133, 18.5356], [19.7255, 14.6383, 9.3120]],
        device=device)
    expected_output = torch.tensor([
        9.6309, 10.8994, 9.6309, 11.9358, 11.9358, 11.9358, 2.8944, 6.2685,
        6.4378, 9.4532, 13.9084, 16.0197, 5.5090, 9.8859, 10.1359, 5.9359,
        8.9269, 13.2211, 12.0422, 12.0422, 12.0422, 6.1559, 13.2413, 14.7687,
        4.4814, 8.4473, 11.6966, 4.2797, 5.3096, 5.6758, 5.1621, 7.0261,
        8.1355, 1.4495, 7.4930, 8.6191
    ],
                                   device=device)
    xyz_batch_cnt = torch.tensor([8, 12], device=device).int()
    new_xyz_batch_cnt = torch.tensor([4, 8], device=device).int()
    max_neighbour_distance = 4.8
    new_xyz_grid_centers = get_dense_voxels_by_center(new_xyz,
                                                      max_neighbour_distance,
                                                      (3, 3, 3))
    nsample = -1
    neighbor_type = 0
    avg_length_of_neighbor_idxs = 1000
    num_total_grids = 27
    neighbor_distance_multiplier = 2.0

    dist, idx, avg_length = three_nn_vector_pool_by_two_step(
        xyz, xyz_batch_cnt, new_xyz, new_xyz_grid_centers, new_xyz_batch_cnt,
        max_neighbour_distance, nsample, neighbor_type,
        avg_length_of_neighbor_idxs, num_total_grids,
        neighbor_distance_multiplier)
    assert (dist[idx != -1].int() == expected_output.int()).all()
