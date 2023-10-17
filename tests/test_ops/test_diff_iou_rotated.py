# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.ops import diff_iou_rotated_2d, diff_iou_rotated_3d
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE

if IS_MLU_AVAILABLE:
    torch.backends.mlu.matmul.allow_tf32 = False


@pytest.mark.parametrize('device', [
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(
            not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
    pytest.param(
        'mlu',
        marks=pytest.mark.skipif(
            not IS_MLU_AVAILABLE, reason='requires MLU support'))
])
def test_diff_iou_rotated_2d(device):
    np_boxes1 = np.asarray([[[0.5, 0.5, 1., 1., .0], [0.5, 0.5, 1., 1., .0],
                             [0.5, 0.5, 1., 1., .0], [0.5, 0.5, 1., 1., .0],
                             [0.5, 0.5, 1., 1., .0]]],
                           dtype=np.float32)
    np_boxes2 = np.asarray(
        [[[0.5, 0.5, 1., 1., .0], [0.5, 0.5, 1., 1., np.pi / 2],
          [0.5, 0.5, 1., 1., np.pi / 4], [1., 1., 1., 1., .0],
          [1.5, 1.5, 1., 1., .0]]],
        dtype=np.float32)

    boxes1 = torch.from_numpy(np_boxes1).to(device)
    boxes2 = torch.from_numpy(np_boxes2).to(device)

    np_expect_ious = np.asarray([[1., 1., .7071, 1 / 7, .0]])
    ious = diff_iou_rotated_2d(boxes1, boxes2)
    assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)


@pytest.mark.parametrize('device', [
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(
            not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
    pytest.param(
        'mlu',
        marks=pytest.mark.skipif(
            not IS_MLU_AVAILABLE, reason='requires MLU support'))
])
def test_diff_iou_rotated_3d(device):
    np_boxes1 = np.asarray(
        [[[.5, .5, .5, 1., 1., 1., .0], [.5, .5, .5, 1., 1., 1., .0],
          [.5, .5, .5, 1., 1., 1., .0], [.5, .5, .5, 1., 1., 1., .0],
          [.5, .5, .5, 1., 1., 1., .0]]],
        dtype=np.float32)
    np_boxes2 = np.asarray(
        [[[.5, .5, .5, 1., 1., 1., .0], [.5, .5, .5, 1., 1., 2., np.pi / 2],
          [.5, .5, .5, 1., 1., 1., np.pi / 4], [1., 1., 1., 1., 1., 1., .0],
          [-1.5, -1.5, -1.5, 2.5, 2.5, 2.5, .0]]],
        dtype=np.float32)

    boxes1 = torch.from_numpy(np_boxes1).to(device)
    boxes2 = torch.from_numpy(np_boxes2).to(device)

    np_expect_ious = np.asarray([[1., .5, .7071, 1 / 15, .0]])
    ious = diff_iou_rotated_3d(boxes1, boxes2)
    assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)
