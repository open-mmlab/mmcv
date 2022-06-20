# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.ops import diff_iou_rotated_2d, diff_iou_rotated_3d

np_boxes_2d_1 = np.asarray(
    [[[0.5, 0.5, 1., 1., .0], [0.5, 0.5, 1., 1., .0], [0.5, 0.5, 1., 1., .0],
      [0.5, 0.5, 1., 1., .0], [0.5, 0.5, 1., 1., .0]]],
    dtype=np.float32)
np_boxes_2d_2 = np.asarray(
    [[[0.5, 0.5, 1., 1., .0], [0.5, 0.5, 1., 1., np.pi / 2],
      [0.5, 0.5, 1., 1., np.pi / 4], [1., 1., 1., 1., .0],
      [1.5, 1.5, 1., 1., .0]]],
    dtype=np.float32)
np_expect_ious_2d = np.asarray([[1., 1., .7071, 1 / 7, .0]])

np_boxes_3d_1 = np.asarray(
    [[[.5, .5, .5, 1., 1., 1., .0], [.5, .5, .5, 1., 1., 1., .0],
      [.5, .5, .5, 1., 1., 1., .0], [.5, .5, .5, 1., 1., 1., .0],
      [.5, .5, .5, 1., 1., 1., .0]]],
    dtype=np.float32)
np_boxes_3d_2 = np.asarray(
    [[[.5, .5, .5, 1., 1., 1., .0], [.5, .5, .5, 1., 1., 2., np.pi / 2],
      [.5, .5, .5, 1., 1., 1., np.pi / 4], [1., 1., 1., 1., 1., 1., .0],
      [-1.5, -1.5, -1.5, 2.5, 2.5, 2.5, .0]]],
    dtype=np.float32)
np_expect_ious_3d = np.asarray([[1., .5, .7071, 1 / 15, .0]])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_diff_iou_rotated_2d_gpu():
    boxes1 = torch.from_numpy(np_boxes_2d_1).cuda()
    boxes2 = torch.from_numpy(np_boxes_2d_2).cuda()
    ious = diff_iou_rotated_2d(boxes1, boxes2)
    assert np.allclose(ious.cpu().numpy(), np_expect_ious_2d, atol=1e-4)


def test_diff_iou_rotated_2d_cpu():
    boxes1 = torch.from_numpy(np_boxes_2d_1)
    boxes2 = torch.from_numpy(np_boxes_2d_2)
    ious = diff_iou_rotated_2d(boxes1, boxes2)
    assert np.allclose(ious.numpy(), np_expect_ious_2d, atol=1e-4)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_diff_iou_rotated_3d_gpu():
    boxes1 = torch.from_numpy(np_boxes_3d_1).cuda()
    boxes2 = torch.from_numpy(np_boxes_3d_2).cuda()
    ious = diff_iou_rotated_3d(boxes1, boxes2)
    assert np.allclose(ious.cpu().numpy(), np_expect_ious_3d, atol=1e-4)


def test_diff_iou_rotated_3d_cpu():
    boxes1 = torch.from_numpy(np_boxes_3d_1)
    boxes2 = torch.from_numpy(np_boxes_3d_2)
    ious = diff_iou_rotated_3d(boxes1, boxes2)
    assert np.allclose(ious.numpy(), np_expect_ious_3d, atol=1e-4)
