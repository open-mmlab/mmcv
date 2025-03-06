# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmcv.ops.pure_pytorch_min_area_polygons.min_area_polygons import min_area_polygons_pytorch


def min_area_polygons(pointsets: torch.Tensor) -> torch.Tensor:
    """Find the smallest polygons that surrounds all points in the point sets.

    Args:
        pointsets (Tensor): point sets with shape  (N, 18).

    Returns:
        torch.Tensor: Return the smallest polygons with shape (N, 8).
    """
    polygons = pointsets.new_zeros((pointsets.size(0), 8))
    min_area_polygons_pytorch(pointsets, polygons)
    return polygons
