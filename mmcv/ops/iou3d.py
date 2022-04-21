# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', [
    'iou3d_boxes_iou3d_forward', 'iou3d_nms3d_forward',
    'iou3d_nms3d_normal_forward'
])


def boxes_iou3d(boxes_a, boxes_b):
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


def nms3d(boxes, scores, iou_threshold):
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


def nms3d_normal(boxes, scores, iou_threshold):
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
