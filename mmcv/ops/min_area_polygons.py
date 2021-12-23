# Copyright (c) OpenMMLab. All rights reserved.
from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['min_area_polygons'])


def min_area_polygons(pointsets):
    """Find the smallest polygons that surrounds all points in the point sets.

    Args:
        pointsets (Tensor):

    Returns:
        torch.Tensor: Return the rotated boxes with shape ().
    """
    polygons = pointsets.new_zeros((pointsets.size(0) * 8))
    ext_module.min_rotated_boxes(pointsets, polygons)
    return polygons.view(-1, 8)
