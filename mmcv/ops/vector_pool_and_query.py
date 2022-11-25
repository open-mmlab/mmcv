from typing import Any, Tuple

import torch
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['stack_vector_pool_forward', 'stack_vector_pool_backward'])


class VectorPoolWithVoxelQuery(Function):

    @staticmethod
    def forward(ctx,
                support_xyz: torch.Tensor,
                xyz_batch_cnt: torch.Tensor,
                support_features: torch.Tensor,
                new_xyz: torch.Tensor,
                new_xyz_batch_cnt: torch.Tensor,
                num_grid_x,
                num_grid_y,
                num_grid_z,
                max_neighbour_distance,
                num_c_out_each_grid,
                use_xyz,
                num_mean_points_per_grid=100,
                nsample=-1,
                neighbor_type=0,
                pooling_type=0):
        """
        Args:
            ctx:
            support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            support_features: (N1 + N2 ..., C)
            new_xyz: (M1 + M2 ..., 3) centers of new positions
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            num_grid_x: number of grids in each local area centered at new_xyz
            num_grid_y:
            num_grid_z:
            max_neighbour_distance:
            num_c_out_each_grid:
            use_xyz:
            neighbor_type: 1: ball, others: cube:
            pooling_type: 0: avg_pool, 1: random choice
        Returns:
            new_features: (M1 + M2 ..., num_c_out)
        """
        assert support_xyz.is_contiguous()
        assert support_features.is_contiguous()
        assert xyz_batch_cnt.is_contiguous()
        assert new_xyz.is_contiguous()
        assert new_xyz_batch_cnt.is_contiguous()
        num_total_grids = num_grid_x * num_grid_y * num_grid_z
        num_c_out = num_c_out_each_grid * num_total_grids
        N, num_c_in = support_features.shape
        M = new_xyz.shape[0]

        assert num_c_in % num_c_out_each_grid == 0, \
            f'the input channels ({num_c_in}) should be an integral multiple of num_c_out_each_grid({num_c_out_each_grid})'

        while True:
            new_features = support_features.new_zeros((M, num_c_out))
            new_local_xyz = support_features.new_zeros(
                (M, 3 * num_total_grids))
            point_cnt_of_grid = xyz_batch_cnt.new_zeros((M, num_total_grids))

            num_max_sum_points = num_mean_points_per_grid * M
            grouped_idxs = xyz_batch_cnt.new_zeros((num_max_sum_points, 3))

            num_cum_sum = ext_module.stack_vector_pool_forward(
                support_xyz, xyz_batch_cnt, support_features, new_xyz,
                new_xyz_batch_cnt, new_features, new_local_xyz,
                point_cnt_of_grid, grouped_idxs, num_grid_x, num_grid_y,
                num_grid_z, max_neighbour_distance, use_xyz,
                num_max_sum_points, nsample, neighbor_type, pooling_type)
            num_mean_points_per_grid = num_cum_sum // M + int(
                num_cum_sum % M > 0)
            if num_cum_sum <= num_max_sum_points:
                break

        grouped_idxs = grouped_idxs[:num_cum_sum]

        normalizer = torch.clamp_min(
            point_cnt_of_grid[:, :, None].float(), min=1e-6)
        new_features = (
            new_features.view(-1, num_total_grids, num_c_out_each_grid) /
            normalizer).view(-1, num_c_out)

        if use_xyz:
            new_local_xyz = (new_local_xyz.view(-1, num_total_grids, 3) /
                             normalizer).view(-1, num_total_grids * 3)

        num_mean_points_per_grid = torch.Tensor([num_mean_points_per_grid
                                                 ]).int()
        nsample = torch.Tensor([nsample]).int()
        ctx.vector_pool_for_backward = (point_cnt_of_grid, grouped_idxs, N,
                                        num_c_in)
        ctx.mark_non_differentiable(new_local_xyz, num_mean_points_per_grid,
                                    nsample, point_cnt_of_grid)
        return new_features, new_local_xyz, num_mean_points_per_grid, point_cnt_of_grid

    @staticmethod
    def backward(ctx, grad_new_features: torch.Tensor,
                 grad_local_xyz: torch.Tensor, grad_num_cum_sum,
                 grad_point_cnt_of_grid):
        """
        Args:
            ctx:
            grad_new_features: (M1 + M2 ..., num_c_out), num_c_out = num_c_out_each_grid * num_total_grids

        Returns:
            grad_support_features: (N1 + N2 ..., C_in)
        """
        point_cnt_of_grid, grouped_idxs, N, num_c_in = ctx.vector_pool_for_backward
        grad_support_features = grad_new_features.new_zeros((N, num_c_in))

        if grouped_idxs.shape[0] > 0:
            ext_module.stack_vector_pool_backward(
                grad_new_features.contiguous(), point_cnt_of_grid,
                grouped_idxs, grad_support_features)

        return None, None, grad_support_features, None, None, None, None, None, None, None, None, None, None, None, None


vector_pool_with_voxel_query = VectorPoolWithVoxelQuery.apply
