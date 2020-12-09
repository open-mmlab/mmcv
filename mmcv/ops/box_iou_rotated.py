from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['box_iou_rotated'])


def box_iou_rotated(bboxes1, bboxes2, aligned=False):
    """Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in
    (x_center, y_center, width, height, angle) format.

    If ``aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Arguments:
        boxes1 (Tensor): rotated bboxes 1. \
            It has shape (N, 5), indicating (x, y, w, h, theta) for each row.
        boxes2 (Tensor): rotated bboxes 2. \
            It has shape (M, 5), indicating (x, y, w, h, theta) for each row.

    Returns:
        ious(Tensor): shape (N, M) if aligned == False else shape (N,)
    """
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if aligned:
        ious = bboxes1.new_zeros(rows)
    else:
        ious = bboxes1.new_zeros((rows * cols))
    bboxes1 = bboxes1.contiguous()
    bboxes2 = bboxes2.contiguous()
    ext_module.box_iou_rotated(bboxes1, bboxes2, ious, aligned=aligned)
    if not aligned:
        ious = ious.view(rows, cols)
    return ious
