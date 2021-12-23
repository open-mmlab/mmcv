# Copyright (c) OpenMMLab. All rights reserved.
from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['convex_iou', 'convex_giou'])


def convex_giou(pointsets, polygons):
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
    convex_giou_grad = ext_module.convex_giou_cuda.convex_giou(
        pointsets, polygons)
    convex_giou_grad = convex_giou_grad.reshape(-1, 19)
    convex_giou = convex_giou_grad[:, -1]
    points_grad = convex_giou_grad[:, 0:-1]
    return convex_giou, points_grad


def convex_iou(pointsets, polygons):
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
    convex_ious = ext_module.convex_iou_cuda.convex_iou(pointsets, polygons)
    convex_ious = convex_ious.reshape(N, K)
    return convex_ious
