import torch

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['box_iou_rotated'])


def box_iou_rotated(bboxes1, bboxes2):
    """Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in
    (x_center, y_center, width, height, angle) format.

    Arguments:
        boxes1 (Tensor[N, 5])
        boxes2 (Tensor[M, 5])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    if torch.__version__ == 'parrots':
        out = torch.zeros((bboxes1.shape[0], bboxes2.shape[0]),
                          dtype=torch.float32).to(bboxes1.device)
        ext_module.box_iou_rotated(bboxes1, bboxes2, out)
    else:
        out = ext_module.box_iou_rotated(bboxes1, bboxes2)
    return out
