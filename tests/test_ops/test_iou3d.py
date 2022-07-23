# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.ops import boxes_iou3d, boxes_overlap_bev, nms3d, nms3d_normal


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_boxes_overlap_bev():
    np_boxes1 = np.asarray([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0],
                            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0],
                            [3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 0.0]],
                           dtype=np.float32)
    np_boxes2 = np.asarray([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0],
                            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, np.pi / 2],
                            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, np.pi / 4]],
                           dtype=np.float32)
    np_expect_overlaps = np.asarray(
        [[4.0, 4.0, (8 + 8 * 2**0.5) /
          (3 + 2 * 2**0.5)], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
        dtype=np.float32)

    boxes1 = torch.from_numpy(np_boxes1).cuda()
    boxes2 = torch.from_numpy(np_boxes2).cuda()

    # test for 3 boxes
    overlaps = boxes_overlap_bev(boxes1, boxes2)
    assert np.allclose(overlaps.cpu().numpy(), np_expect_overlaps, atol=1e-4)

    # test for many boxes
    boxes2 = boxes2.repeat_interleave(555, 0)

    overlaps = boxes_overlap_bev(boxes1, boxes2)
    assert np.allclose(
        overlaps.cpu().numpy(), np_expect_overlaps.repeat(555, 1), atol=1e-4)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_boxes_iou3d():
    np_boxes1 = np.asarray([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0],
                            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0],
                            [3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 0.0]],
                           dtype=np.float32)
    np_boxes2 = np.asarray([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0],
                            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, np.pi / 2],
                            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, np.pi / 4]],
                           dtype=np.float32)
    np_expect_ious = np.asarray(
        [[1.0, 1.0, 1.0 / 2**0.5], [1.0 / 15, 1.0 / 15, 1.0 / 15],
         [0.0, 0.0, 0.0]],
        dtype=np.float32)

    boxes1 = torch.from_numpy(np_boxes1).cuda()
    boxes2 = torch.from_numpy(np_boxes2).cuda()

    ious = boxes_iou3d(boxes1, boxes2)
    assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_nms3d():
    # test for 5 boxes
    np_boxes = np.asarray([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0],
                           [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0],
                           [3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 0.3],
                           [3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 0.0],
                           [3.0, 3.2, 3.2, 3.0, 2.0, 2.0, 0.3]],
                          dtype=np.float32)
    np_scores = np.array([0.6, 0.9, 0.1, 0.2, 0.15], dtype=np.float32)
    np_inds = np.array([1, 0, 3])
    boxes = torch.from_numpy(np_boxes)
    scores = torch.from_numpy(np_scores)
    inds = nms3d(boxes.cuda(), scores.cuda(), iou_threshold=0.3)

    assert np.allclose(inds.cpu().numpy(), np_inds)

    # test for many boxes
    np.random.seed(42)
    np_boxes = np.random.rand(555, 7).astype(np.float32)
    np_scores = np.random.rand(555).astype(np.float32)
    boxes = torch.from_numpy(np_boxes)
    scores = torch.from_numpy(np_scores)
    inds = nms3d(boxes.cuda(), scores.cuda(), iou_threshold=0.3)

    assert len(inds.cpu().numpy()) == 176


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_nms3d_normal():
    # test for 5 boxes
    np_boxes = np.asarray([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0],
                           [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0],
                           [3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 0.3],
                           [3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 0.0],
                           [3.0, 3.2, 3.2, 3.0, 2.0, 2.0, 0.3]],
                          dtype=np.float32)
    np_scores = np.array([0.6, 0.9, 0.1, 0.2, 0.15], dtype=np.float32)
    np_inds = np.array([1, 0, 3])
    boxes = torch.from_numpy(np_boxes)
    scores = torch.from_numpy(np_scores)
    inds = nms3d_normal(boxes.cuda(), scores.cuda(), iou_threshold=0.3)

    assert np.allclose(inds.cpu().numpy(), np_inds)

    # test for many boxes
    np.random.seed(42)
    np_boxes = np.random.rand(555, 7).astype(np.float32)
    np_scores = np.random.rand(555).astype(np.float32)
    boxes = torch.from_numpy(np_boxes)
    scores = torch.from_numpy(np_scores)
    inds = nms3d_normal(boxes.cuda(), scores.cuda(), iou_threshold=0.3)

    assert len(inds.cpu().numpy()) == 148
