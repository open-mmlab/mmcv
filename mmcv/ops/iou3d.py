# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional

import torch
from torch import Tensor

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', [
    'iou3d_boxes_iou3d_forward', 'iou3d_nms3d_forward',
    'iou3d_nms3d_normal_forward'
])


def boxes_iou3d(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Calculate boxes 3D IoU.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 7).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 7).

    Returns:
        torch.Tensor: IoU result with shape (M, N).
    """
    ans_iou = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))

    ext_module.iou3d_boxes_iou3d_forward(boxes_a.contiguous(),
                                         boxes_b.contiguous(), ans_iou)

    return ans_iou


def nms3d(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """3D NMS function GPU implementation (for BEV boxes).

    Args:
        boxes (torch.Tensor): Input boxes with the shape of (N, 7)
            ([x, y, z, dx, dy, dz, heading]).
        scores (torch.Tensor): Scores of boxes with the shape of (N).
        iou_threshold (float): Overlap threshold of NMS.

    Returns:
        torch.Tensor: Indexes after NMS.
    """
    assert boxes.size(1) == 7, 'Input boxes shape should be (N, 7)'
    order = scores.sort(0, descending=True)[1]
    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = torch.zeros(size=(), dtype=torch.long)
    ext_module.iou3d_nms3d_forward(
        boxes, keep, num_out, nms_overlap_thresh=iou_threshold)
    keep = order[keep[:num_out].cuda(boxes.device)].contiguous()
    return keep


def nms3d_normal(boxes: Tensor, scores: Tensor,
                 iou_threshold: float) -> Tensor:
    """Normal 3D NMS function GPU implementation. The overlap of two boxes for
    IoU calculation is defined as the exact overlapping area of the two boxes
    WITH their yaw angle set to 0.

    Args:
        boxes (torch.Tensor): Input boxes with shape (N, 7).
            ([x, y, z, dx, dy, dz, heading]).
        scores (torch.Tensor): Scores of predicted boxes with shape (N).
        iou_threshold (float): Overlap threshold of NMS.

    Returns:
        torch.Tensor: Remaining indices with scores in descending order.
    """
    assert boxes.shape[1] == 7, 'Input boxes shape should be (N, 7)'
    order = scores.sort(0, descending=True)[1]
    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = torch.zeros(size=(), dtype=torch.long)
    ext_module.iou3d_nms3d_normal_forward(
        boxes, keep, num_out, nms_overlap_thresh=iou_threshold)
    return order[keep[:num_out].cuda(boxes.device)].contiguous()


def _xyxyr2xywhr(boxes: Tensor) -> Tensor:
    """Convert [x1, y1, x2, y2, heading] box to [x, y, dx, dy, heading] box.

    Args:
        box (torch.Tensor): Input boxes with shape (N, 5).

    Returns:
        torch.Tensor: Converted boxes with shape (N, 7).
    """
    warnings.warn(
        'This function is deprecated and will be removed in the future.',
        DeprecationWarning)
    return torch.stack(
        ((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2,
         boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1], boxes[:, 4]),
        dim=-1)


def boxes_iou_bev(boxes_a, boxes_b):
    """Calculate boxes IoU in the Bird's Eye View.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 5)
            ([x1, y1, x2, y2, ry]).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5)
            ([x1, y1, x2, y2, ry]).

    Returns:
        torch.Tensor: IoU result with shape (M, N).
    """
    from .box_iou_rotated import box_iou_rotated

    warnings.warn(
        '`iou3d.boxes_iou_bev` is deprecated and will be removed in'
        ' the future. Please, use `box_iou_rotated.box_iou_rotated`.',
        DeprecationWarning)

    return box_iou_rotated(_xyxyr2xywhr(boxes_a), _xyxyr2xywhr(boxes_b))


def nms_bev(boxes: Tensor,
            scores: Tensor,
            thresh: float,
            pre_max_size: Optional[int] = None,
            post_max_size: Optional[int] = None) -> Tensor:
    """NMS function GPU implementation (for BEV boxes).

    The overlap of two
    boxes for IoU calculation is defined as the exact overlapping area of the
    two boxes. In this function, one can also set ``pre_max_size`` and
    ``post_max_size``.
    Args:
        boxes (torch.Tensor): Input boxes with the shape of (N, 5)
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of (N,).
        thresh (float): Overlap threshold of NMS.
        pre_max_size (int, optional): Max size of boxes before NMS.
            Default: None.
        post_max_size (int, optional): Max size of boxes after NMS.
            Default: None.
    Returns:
        torch.Tensor: Indexes after NMS.
    """
    from .nms import nms_rotated

    warnings.warn(
        '`iou3d.nms_bev` is deprecated and will be removed in'
        ' the future. Please, use `nms.nms_rotated`.', DeprecationWarning)
    assert boxes.size(1) == 5, 'Input boxes shape should be (N, 5)'
    order = scores.sort(0, descending=True)[1]

    if pre_max_size is not None:
        order = order[:pre_max_size]
    boxes = _xyxyr2xywhr(boxes)[order]
    scores = scores[order]

    keep = nms_rotated(boxes, scores, thresh)[1]
    keep = order[keep]

    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


def nms_normal_bev(boxes: Tensor, scores: Tensor, thresh: float) -> Tensor:
    """Normal NMS function GPU implementation (for BEV boxes).

    The overlap of
    two boxes for IoU calculation is defined as the exact overlapping area of
    the two boxes WITH their yaw angle set to 0.
    Args:
        boxes (torch.Tensor): Input boxes with shape (N, 5)
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of predicted boxes with shape (N,).
        thresh (float): Overlap threshold of NMS.
    Returns:
        torch.Tensor: Remaining indices with scores in descending order.
    """
    from .nms import nms

    warnings.warn(
        '`iou3d.nms_normal_bev` is deprecated and will be removed in'
        ' the future. Please, use `nms.nms`.', DeprecationWarning)
    assert boxes.shape[1] == 5, 'Input boxes shape should be (N, 5)'

    return nms(boxes[:, :-1], scores, thresh)[1]
