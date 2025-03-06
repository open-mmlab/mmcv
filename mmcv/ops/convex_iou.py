# Copyright (c) OpenMMLab. All rights reserved.

import torch

from mmcv.ops.pure_pytorch_convex_iou.convex_giou import convex_giou_pytorch
from mmcv.ops.pure_pytorch_convex_iou.convex_iou import convex_iou_pytorch


def convex_giou(pointsets: torch.Tensor,
                polygons: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return generalized intersection-over-union (Jaccard index) between point
    sets and polygons.

    Args:
        pointsets (torch.Tensor): It has shape (N, 18),
            indicating (x1, y1, x2, y2, ..., x9, y9) for each row.
        polygons (torch.Tensor): It has shape (N, 8),
            indicating (x1, y1, x2, y2, x3, y3, x4, y4) for each row.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The first element is the gious
        between point sets and polygons with the shape (N,). The second
        element is the gradient of point sets with the shape (N, 18).
    """
    output = pointsets.new_zeros((pointsets.size(0), 19))
    convex_giou_pytorch(pointsets, polygons, output)
    convex_giou = output[:, -1]
    points_grad = output[:, 0:-1]
    return convex_giou, points_grad


def convex_iou(pointsets: torch.Tensor,
               polygons: torch.Tensor) -> torch.Tensor:
    """Return intersection-over-union (Jaccard index) between point sets and
    polygons.

    Args:
        pointsets (torch.Tensor): It has shape (N, 18),
            indicating (x1, y1, x2, y2, ..., x9, y9) for each row.
        polygons (torch.Tensor): It has shape (K, 8),
            indicating (x1, y1, x2, y2, x3, y3, x4, y4) for each row.

    Returns:
        torch.Tensor: Return the ious between point sets and polygons with the
        shape (N, K).
    """
    N, K = pointsets.size(0), polygons.size(0)
    ious = pointsets.new_zeros((N, K))
    convex_iou_pytorch(pointsets, polygons, ious)
    return ious
