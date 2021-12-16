# Copyright (c) OpenMMLab. All rights reserved.
from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['box_iou_rotated'])


def box_iou_rotated(bboxes1,
                    bboxes2,
                    mode='iou',
                    aligned=False,
                    clockwise=True):
    """Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in
    (x_center, y_center, width, height, angle) format.

    If ``aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    coordinates details:
        the positive direction along x axis is left->right
        the positive direction along y axis is top->down
        the w border is in parallel with x axis when angle = 0
        there are 2 opposite definitions of the positive angular direction,
        CW and CCW. MMCV supports both definitions and by default CW.
        Please set clockwise=False if you are using the CCW definition.

        if clockwise is True

        .. code-block:: none

            0-------------------> x (0 rad)
            |  .-------------.
            |  |             |
            |  |     box     h
            |  |   angle=0   |
            |  .------w------.
            v
            y (pi/2 rad)
            In such coordination system the rotation matrix is:
                [cosa -sina]
                [sina  cosa]

        if clockwise is False

        .. code-block:: none

            0-------------------> x (0 rad)
            |  .-------------.
            |  |             |
            |  |     box     h
            |  |   angle=0   |
            |  .------w------.
            v
            y (-pi/2 rad)
            In such coordination system the rotation matrix is:
                [cosa  sina]
                [-sina cosa]

    Args:
        boxes1 (torch.Tensor): rotated bboxes 1. It has shape (N, 5),
            indicating (x, y, w, h, theta) for each row. Note that theta is in
            radian.
        boxes2 (torch.Tensor): rotated bboxes 2. It has shape (M, 5),
            indicating (x, y, w, h, theta) for each row. Note that theta is in
            radian.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).
        clockwise (bool): flag indicating whether the positive angular
            orientation is clockwise. default True.

    Returns:
        torch.Tensor: Return the ious betweens boxes. If ``aligned`` is
        ``False``, the shape of ious is (N, M) else (N,).
    """
    assert mode in ['iou', 'iof']
    mode_dict = {'iou': 0, 'iof': 1}
    mode_flag = mode_dict[mode]
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if aligned:
        ious = bboxes1.new_zeros(rows)
    else:
        ious = bboxes1.new_zeros((rows * cols))
    bboxes1 = bboxes1.contiguous()
    bboxes2 = bboxes2.contiguous()
    if not clockwise:
        bboxes1 = bboxes1.clone().detach()
        bboxes2 = bboxes2.clone().detach()
        bboxes1[..., -1] *= -1
        bboxes2[..., -1] *= -1
    ext_module.box_iou_rotated(
        bboxes1, bboxes2, ious, mode_flag=mode_flag, aligned=aligned)
    if not aligned:
        ious = ious.view(rows, cols)
    return ious
