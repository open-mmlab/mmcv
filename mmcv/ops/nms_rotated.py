import torch
from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['ml_nms_rotated', 'nms_rotated'])


def ml_nms_rotated(boxes, labels, iou_thr):
    if boxes.shape[0] == 0:
        return boxes
    dets = boxes[:, :5]
    scores = boxes[:, 5]
    if torch.__version__ == 'parrots':
        dets_wl = torch.cat((dets, labels.unsqueeze(1)), 1)
        _, order = scores.sort(0, descending=True)
        dets_sorted = dets_wl.index_select(0, order)
        select = torch.zeros((dets.shape[0]),
                             dtype=torch.int64).to(dets.device)
        ext_module.ml_nms_rotated(
            dets, scores, labels, dets_sorted, select, iou_threshold=iou_thr)
        keep_inds = order.masked_select(select == 1)
        dets = dets[keep_inds, :]
        return dets, keep_inds
    else:
        keep_inds = ext_module.ml_nms_rotated(dets, scores, labels, iou_thr)
        dets = dets[keep_inds, :]
        return dets, keep_inds


def nms_rotated(boxes, iou_thr):
    if boxes.shape[0] == 0:
        return boxes
    dets = boxes[:, :5]
    scores = boxes[:, 5]
    if torch.__version__ == 'parrots':
        _, order = scores.sort(0, descending=True)
        dets_sorted = dets.index_select(0, order)
        select = torch.zeros((dets.shape[0]),
                             dtype=torch.int64).to(dets.device)
        ext_module.nms_rotated(
            dets, scores, dets_sorted, select, iou_threshold=iou_thr)
        keep_inds = order.masked_select(select == 1)
        dets = dets[keep_inds, :]
        return dets, keep_inds
    else:
        keep_inds = ext_module.nms_rotated(dets, scores, iou_thr)
        dets = dets[keep_inds, :]
    return dets, keep_inds
