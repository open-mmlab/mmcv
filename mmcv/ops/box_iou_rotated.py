import torch

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['box_iou_rotated'])


def box_iou_rotated(bboxes1, bboxes2):
    if torch.__version__ == 'parrots':
        out = torch.zeros((bboxes1.shape[0], bboxes2.shape[0]),
                          dtype=torch.float32).to(bboxes1.device)
        ext_module.box_iou_rotated(bboxes1, bboxes2, out)
    else:
        out = ext_module.box_iou_rotated(bboxes1, bboxes2)
    return out
