# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext',
    ['dynamic_point_to_voxel_forward', 'dynamic_point_to_voxel_backward'])


class _DynamicScatter(Function):

    @staticmethod
    def forward(ctx: Any,
                feats: torch.Tensor,
                coors: torch.Tensor,
                reduce_type: str = 'max') -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert kitti points(N, >=3) to voxels.

        Args:
            feats (torch.Tensor): [N, C]. Points features to be reduced
                into voxels.
            coors (torch.Tensor): [N, ndim]. Corresponding voxel coordinates
                (specifically multi-dim voxel index) of each points.
            reduce_type (str, optional): Reduce op. support 'max', 'sum' and
                'mean'. Default: 'max'.

        Returns:
            tuple[torch.Tensor]: A tuple contains two elements. The first one
            is the voxel features with shape [M, C] which are respectively
            reduced from input features that share the same voxel coordinates.
            The second is voxel coordinates with shape [M, ndim].
        """
        ctx.device = feats.device.type
        if ctx.device == 'npu':
            import mx_driving._C
            voxel_idx = mx_driving._C.point_to_voxel(coors, [], [], 'XYZ')
            unique_res = mx_driving._C.unique_voxel(voxel_idx)
            num_voxels, uniqued_voxel_idx, prefix_sum, \
                argsort_coor, _ = unique_res
            voxel_coors = \
                mx_driving._C.voxel_to_point(uniqued_voxel_idx, [], [], 'XYZ')
            voxel_feats, \
                compare_mask = mx_driving._C.npu_dynamic_scatter(feats, coors,
                                                         prefix_sum,
                                                         argsort_coor,
                                                         num_voxels,
                                                         reduce_type)
            ctx.reduce_type = reduce_type
            ctx.feats_shape = feats.shape
            ctx.save_for_backward(prefix_sum, argsort_coor, compare_mask)
            ctx.mark_non_differentiable(voxel_coors)
            return voxel_feats, voxel_coors

        results = ext_module.dynamic_point_to_voxel_forward(
            feats, coors, reduce_type)
        voxel_feats, voxel_coors, point2voxel_map, voxel_points_count = results
        ctx.reduce_type = reduce_type
        ctx.save_for_backward(feats, voxel_feats, point2voxel_map,
                              voxel_points_count)
        ctx.mark_non_differentiable(voxel_coors)
        return voxel_feats, voxel_coors

    @staticmethod
    def backward(ctx: Any,
                 grad_voxel_feats: torch.Tensor,
                 grad_voxel_coors: Optional[torch.Tensor] = None) -> tuple:
        if ctx.device == 'npu':
            import mx_driving._C
            prefix_sum, argsort_coor, compare_mask = ctx.saved_tensors
            grad_point_feats = torch.zeros(
                ctx.feats_shape,
                dtype=grad_voxel_feats.dtype,
                device=grad_voxel_feats.device)
            mx_driving._C.npu_dynamic_scatter_grad(grad_point_feats,
                                           grad_voxel_feats.contiguous(),
                                           prefix_sum, argsort_coor,
                                           compare_mask, ctx.reduce_type)
            return grad_point_feats, None, None

        (feats, voxel_feats, point2voxel_map,
         voxel_points_count) = ctx.saved_tensors
        grad_feats = torch.zeros_like(feats)
        # TODO: whether to use index put or use cuda_backward
        # To use index put, need point to voxel index
        ext_module.dynamic_point_to_voxel_backward(
            grad_feats, grad_voxel_feats.contiguous(), feats, voxel_feats,
            point2voxel_map, voxel_points_count, ctx.reduce_type)
        return grad_feats, None, None


dynamic_scatter = _DynamicScatter.apply


class DynamicScatter(nn.Module):
    """Scatters points into voxels, used in the voxel encoder with dynamic
    voxelization.

    Note:
        The CPU and GPU implementation get the same output, but have numerical
        difference after summation and division (e.g., 5e-7).

    Args:
        voxel_size (list): list [x, y, z] size of three dimension.
        point_cloud_range (list): The coordinate range of points, [x_min,
            y_min, z_min, x_max, y_max, z_max].
        average_points (bool): whether to use avg pooling to scatter points
            into voxel.
    """

    def __init__(self, voxel_size: List, point_cloud_range: List,
                 average_points: bool):
        super().__init__()

        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.average_points = average_points

    def forward_single(
            self, points: torch.Tensor,
            coors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scatters points into voxels.

        Args:
            points (torch.Tensor): Points to be reduced into voxels.
            coors (torch.Tensor): Corresponding voxel coordinates (specifically
                multi-dim voxel index) of each points.

        Returns:
            tuple[torch.Tensor]: A tuple contains two elements. The first one
            is the voxel features with shape [M, C] which are respectively
            reduced from input features that share the same voxel coordinates.
            The second is voxel coordinates with shape [M, ndim].
        """
        reduce = 'mean' if self.average_points else 'max'
        return dynamic_scatter(points.contiguous(), coors.contiguous(), reduce)

    def forward(self, points: torch.Tensor,
                coors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scatters points/features into voxels.

        Args:
            points (torch.Tensor): Points to be reduced into voxels.
            coors (torch.Tensor): Corresponding voxel coordinates (specifically
                multi-dim voxel index) of each points.

        Returns:
            tuple[torch.Tensor]: A tuple contains two elements. The first one
            is the voxel features with shape [M, C] which are respectively
            reduced from input features that share the same voxel coordinates.
            The second is voxel coordinates with shape [M, ndim].
        """
        if coors.size(-1) == 3:
            return self.forward_single(points, coors)
        else:
            batch_size = coors[-1, 0] + 1
            voxels, voxel_coors = [], []
            for i in range(batch_size):
                inds = torch.where(coors[:, 0] == i)
                voxel, voxel_coor = self.forward_single(
                    points[inds], coors[inds][:, 1:])
                coor_pad = F.pad(voxel_coor, (1, 0), mode='constant', value=i)
                voxel_coors.append(coor_pad)
                voxels.append(voxel)
            features = torch.cat(voxels, dim=0)
            feature_coors = torch.cat(voxel_coors, dim=0)

            return features, feature_coors

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'voxel_size=' + str(self.voxel_size)
        s += ', point_cloud_range=' + str(self.point_cloud_range)
        s += ', average_points=' + str(self.average_points)
        s += ')'
        return s
