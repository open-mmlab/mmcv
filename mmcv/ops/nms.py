import sys

import numpy as np
import torch

from mmcv.utils import deprecated_api_warning
from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['nms', 'softnms', 'nms_match'])


# This function is modified from: https://github.com/pytorch/vision/
class NMSop(torch.autograd.Function):

    @staticmethod
    def forward(ctx, bboxes, scores, iou_threshold, offset):
        inds = ext_module.nms(
            bboxes, scores, iou_threshold=float(iou_threshold), offset=offset)
        return inds

    @staticmethod
    def symbolic(g, bboxes, scores, iou_threshold, offset):
        from torch.onnx.symbolic_opset9 import select, squeeze, unsqueeze

        boxes = unsqueeze(g, bboxes, 0)
        scores = unsqueeze(g, unsqueeze(g, scores, 0), 0)
        max_output_per_class = g.op(
            'Constant', value_t=torch.tensor([sys.maxsize], dtype=torch.long))
        iou_threshold = g.op(
            'Constant',
            value_t=torch.tensor([iou_threshold], dtype=torch.float))
        nms_out = g.op('NonMaxSuppression', boxes, scores,
                       max_output_per_class, iou_threshold)
        return squeeze(
            g,
            select(
                g, nms_out, 1,
                g.op('Constant', value_t=torch.tensor([2], dtype=torch.long))),
            1)


@deprecated_api_warning({'iou_thr': 'iou_threshold'})
def nms(boxes, scores, iou_threshold, offset=0):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either torch tensor or numpy array. GPU NMS will be used
    if the input is gpu tensor, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        boxes (torch.Tensor or np.ndarray): boxes in shape (N, 4).
        scores (torch.Tensor or np.ndarray): scores in shape (N, ).
        iou_threshold (float): IoU threshold for NMS.
        offset (int, 0 or 1): boxes' width or height is (x2 - x1 + offset).

    Returns:
        tuple: kept dets(boxes and scores) and indice, which is always the \
            same data type as the input.

    Example:
        >>> boxes = np.array([[49.1, 32.4, 51.0, 35.9],
        >>>                   [49.3, 32.9, 51.0, 35.3],
        >>>                   [49.2, 31.8, 51.0, 35.4],
        >>>                   [35.1, 11.5, 39.1, 15.7],
        >>>                   [35.6, 11.8, 39.3, 14.2],
        >>>                   [35.3, 11.5, 39.9, 14.5],
        >>>                   [35.2, 11.7, 39.7, 15.7]], dtype=np.float32)
        >>> scores = np.array([0.9, 0.9, 0.5, 0.5, 0.5, 0.4, 0.3],\
               dtype=np.float32)
        >>> iou_threshold = 0.6
        >>> dets, inds = nms(boxes, scores, iou_threshold)
        >>> assert len(inds) == len(dets) == 3
    """
    assert isinstance(boxes, (torch.Tensor, np.ndarray))
    assert isinstance(scores, (torch.Tensor, np.ndarray))
    is_numpy = False
    if isinstance(boxes, np.ndarray):
        is_numpy = True
        boxes = torch.from_numpy(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    assert boxes.size(1) == 4
    assert boxes.size(0) == scores.size(0)
    assert offset in (0, 1)

    if torch.__version__ == 'parrots':
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + offset) * (y2 - y1 + offset)
        _, order = scores.sort(0, descending=True)
        if boxes.device == 'cpu':
            indata_list = [boxes, order, areas]
            indata_dict = {
                'iou_threshold': float(iou_threshold),
                'offset': int(offset)
            }
            select = ext_module.nms(*indata_list, **indata_dict).byte()
        else:
            boxes_sorted = boxes.index_select(0, order)
            indata_list = [boxes_sorted, order, areas]
            indata_dict = {
                'iou_threshold': float(iou_threshold),
                'offset': int(offset)
            }
            select = ext_module.nms(*indata_list, **indata_dict)
        inds = order.masked_select(select)
    else:
        if torch.onnx.is_in_onnx_export() and offset == 0:
            # ONNX only support offset == 1
            boxes[:, -2:] -= 1
        inds = NMSop.apply(boxes, scores, iou_threshold, offset)
    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
    if is_numpy:
        dets = dets.cpu().numpy()
        inds = inds.cpu().numpy()
    return dets, inds


@deprecated_api_warning({'iou_thr': 'iou_threshold'})
def soft_nms(boxes,
             scores,
             iou_threshold=0.3,
             sigma=0.5,
             min_score=1e-3,
             method='linear',
             offset=0):
    """Dispatch to only CPU Soft NMS implementations.

    The input can be either a torch tensor or numpy array.
    The returned type will always be the same as inputs.

    Arguments:
        boxes (torch.Tensor or np.ndarray): boxes in shape (N, 4).
        scores (torch.Tensor or np.ndarray): scores in shape (N, ).
        iou_threshold (float): IoU threshold for NMS.
        sigma (float): hyperparameter for gaussian method
        min_score (float): score filter threshold
        method (str): either 'linear' or 'gaussian'
        offset (int, 0 or 1): boxes' width or height is (x2 - x1 + offset).

    Returns:
        tuple: kept dets(boxes and scores) and indice, which is always the \
            same data type as the input.

    Example:
        >>> boxes = np.array([[4., 3., 5., 3.],
        >>>                   [4., 3., 5., 4.],
        >>>                   [3., 1., 3., 1.],
        >>>                   [3., 1., 3., 1.],
        >>>                   [3., 1., 3., 1.],
        >>>                   [3., 1., 3., 1.]], dtype=np.float32)
        >>> scores = np.array([0.9, 0.9, 0.5, 0.5, 0.4, 0.0], dtype=np.float32)
        >>> iou_threshold = 0.6
        >>> dets, inds = soft_nms(boxes, scores, iou_threshold, sigma=0.5)
        >>> assert len(inds) == len(dets) == 5
    """

    assert isinstance(boxes, (torch.Tensor, np.ndarray))
    assert isinstance(scores, (torch.Tensor, np.ndarray))
    is_numpy = False
    if isinstance(boxes, np.ndarray):
        is_numpy = True
        boxes = torch.from_numpy(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    assert boxes.size(1) == 4
    assert boxes.size(0) == scores.size(0)
    assert offset in (0, 1)
    method_dict = {'naive': 0, 'linear': 1, 'gaussian': 2}
    assert method in method_dict.keys()

    if torch.__version__ == 'parrots':
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + offset) * (y2 - y1 + offset)
        indata_list = [boxes.cpu(), scores.cpu(), areas.cpu()]
        indata_dict = {
            'iou_threshold': float(iou_threshold),
            'sigma': float(sigma),
            'min_score': min_score,
            'method': method_dict[method],
            'offset': int(offset)
        }
        dets, inds, num_out = ext_module.softnms(*indata_list, **indata_dict)
        inds = inds[:num_out]
    else:
        dets = boxes.new_empty((boxes.size(0), 5), device='cpu')
        inds = ext_module.softnms(
            boxes.cpu(),
            scores.cpu(),
            dets.cpu(),
            iou_threshold=float(iou_threshold),
            sigma=float(sigma),
            min_score=float(min_score),
            method=method_dict[method],
            offset=int(offset))
    dets = dets[:inds.size(0)]
    if is_numpy:
        dets = dets.cpu().numpy()
        inds = inds.cpu().numpy()
        return dets, inds
    else:
        return dets.to(device=boxes.device), inds.to(device=boxes.device)


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.

    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict): specify nms type and other parameters like iou_thr.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class

    Returns:
        tuple: kept dets and indice.
    """
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)
    dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
    boxes = boxes[keep]
    scores = dets[:, -1]
    return torch.cat([boxes, scores[:, None]], -1), keep


def nms_match(dets, iou_threshold):
    """Matched dets into different groups by NMS.

    NMS match is Similar to NMS but when a bbox is suppressed, nms match will
    record the indice of suppressed bbox and form a group with the indice of
    kept bbox. In each group, indice is sorted as score order.

    Arguments:
        dets (torch.Tensor | np.ndarray): Det boxes with scores, shape (N, 5).
        iou_thr (float): IoU thresh for NMS.

    Returns:
        List[torch.Tensor | np.ndarray]: The outer list corresponds different
            matched group, the inner Tensor corresponds the indices for a group
            in score order.
    """
    if dets.shape[0] == 0:
        matched = []
    else:
        assert dets.shape[-1] == 5, 'inputs dets.shape should be (N, 5), ' \
                                    f'but get {dets.shape}'
        if isinstance(dets, torch.Tensor):
            dets_t = dets.detach().cpu()
        else:
            dets_t = torch.from_numpy(dets)
        matched = ext_module.nms_match(dets_t, float(iou_threshold))

    if isinstance(dets, torch.Tensor):
        return [dets.new_tensor(m, dtype=torch.long) for m in matched]
    else:
        return [np.array(m, dtype=np.int) for m in matched]
