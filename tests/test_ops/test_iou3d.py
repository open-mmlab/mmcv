import numpy as np
import pytest
import torch

from mmcv.ops import boxes_iou_bev, nms_gpu


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_boxes_iou_bev():
    np_boxes1 = np.asarray(
        [[1.0, 1.0, 3.0, 4.0, 0.5], [2.0, 2.0, 3.0, 4.0, 0.6],
         [7.0, 7.0, 8.0, 8.0, 0.4]],
        dtype=np.float32)
    np_boxes2 = np.asarray(
        [[0.0, 2.0, 2.0, 5.0, 0.3], [2.0, 1.0, 3.0, 3.0, 0.5],
         [5.0, 5.0, 6.0, 7.0, 0.4]],
        dtype=np.float32)
    np_expect_ious = np.asarray(
        [[0.3708, 0.4351, 0.0000], [0.1104, 0.4487, 0.0424],
         [0.0000, 0.0000, 0.3622]],
        dtype=np.float32)

    boxes1 = torch.from_numpy(np_boxes1).cuda()
    boxes2 = torch.from_numpy(np_boxes2).cuda()

    ious = boxes_iou_bev(boxes1, boxes2)
    assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_nms_gpu():
    np_boxes = np.array([[6.0, 3.0, 8.0, 7.0], [3.0, 6.0, 9.0, 11.0],
                         [3.0, 7.0, 10.0, 12.0], [1.0, 4.0, 13.0, 7.0]],
                        dtype=np.float32)
    np_scores = np.array([0.6, 0.9, 0.7, 0.2], dtype=np.float32)
    np_inds = np.array([1, 0, 3])
    np_dets = np.array([[3.0, 6.0, 9.0, 11.0, 0.9], [6.0, 3.0, 8.0, 7.0, 0.6],
                        [1.0, 4.0, 13.0, 7.0, 0.2]])
    boxes = torch.from_numpy(np_boxes)
    scores = torch.from_numpy(np_scores)
    dets, inds = nms_gpu(boxes.cuda(), scores.cuda(), iou_threshold=0.3)

    assert np.allclose(dets.cpu().numpy(), np_dets)  # test gpu
    assert np.allclose(inds.cpu().numpy(), np_inds)  # test gpu
