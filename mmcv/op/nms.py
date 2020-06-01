import torch

from ..utils import ext_loader

ext_module = ext_loader.load_ext('op_ext', ['nms', 'softnms'])


def nms(boxes, scores, iou_threshold, offset=0):
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
        inds = ext_module.nms(boxes, scores, float(iou_threshold), int(offset))
    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
    return dets, inds


def soft_nms(boxes,
             scores,
             iou_threshold=0.3,
             sigma=0.5,
             min_score=1e-3,
             method='linear',
             offset=0):
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
        inds = ext_module.softnms(boxes.cpu(), scores.cpu(), dets.cpu(),
                                  float(iou_threshold), float(sigma),
                                  float(min_score), method_dict[method],
                                  int(offset))
    dets = dets[:inds.size(0)]
    return dets.to(device=boxes.device), inds.to(device=boxes.device)


def batched_nms(boxes, scores, idxs, nms_cfg):
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

    Returns:
        tuple: kept dets and indice.
    """
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)
    dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
    boxes = boxes[keep]
    scores = dets[:, -1]
    return torch.cat([boxes, scores[:, None]], -1), keep
