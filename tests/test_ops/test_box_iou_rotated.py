# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.ops import box_iou_rotated
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE, IS_NPU_AVAILABLE


class TestBoxIoURotated:

    def test_box_iou_rotated_cpu(self):
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
        np_expect_ious_aligned = np.asarray([0.3708, 0.4487, 0.3622],
                                            dtype=np.float32)

        boxes1 = torch.from_numpy(np_boxes1)
        boxes2 = torch.from_numpy(np_boxes2)

        # test cw angle definition
        ious = box_iou_rotated(boxes1, boxes2)
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)

        ious = box_iou_rotated(boxes1, boxes2, aligned=True)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-4)

        # test ccw angle definition
        boxes1[..., -1] *= -1
        boxes2[..., -1] *= -1
        ious = box_iou_rotated(boxes1, boxes2, clockwise=False)
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)

        ious = box_iou_rotated(boxes1, boxes2, aligned=True, clockwise=False)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-4)

    @pytest.mark.parametrize('device', [
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
        pytest.param(
            'mlu',
            marks=pytest.mark.skipif(
                not IS_MLU_AVAILABLE, reason='requires MLU support')),
        pytest.param(
            'npu',
            marks=pytest.mark.skipif(
                not IS_NPU_AVAILABLE, reason='requires NPU support'))
    ])
    def test_box_iou_rotated(self, device):
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
        np_expect_ious_aligned = np.asarray([0.3708, 0.4487, 0.3622],
                                            dtype=np.float32)

        boxes1 = torch.from_numpy(np_boxes1).to(device)
        boxes2 = torch.from_numpy(np_boxes2).to(device)

        # test cw angle definition
        ious = box_iou_rotated(boxes1, boxes2)
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)

        ious = box_iou_rotated(boxes1, boxes2, aligned=True)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-4)

        # test ccw angle definition
        boxes1[..., -1] *= -1
        boxes2[..., -1] *= -1
        ious = box_iou_rotated(boxes1, boxes2, clockwise=False)
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)

        ious = box_iou_rotated(boxes1, boxes2, aligned=True, clockwise=False)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-4)

    def test_box_iou_rotated_iof_cpu(self):
        np_boxes1 = np.asarray(
            [[1.0, 1.0, 3.0, 4.0, 0.5], [2.0, 2.0, 3.0, 4.0, 0.6],
             [7.0, 7.0, 8.0, 8.0, 0.4]],
            dtype=np.float32)
        np_boxes2 = np.asarray(
            [[0.0, 2.0, 2.0, 5.0, 0.3], [2.0, 1.0, 3.0, 3.0, 0.5],
             [5.0, 5.0, 6.0, 7.0, 0.4]],
            dtype=np.float32)
        np_expect_ious = np.asarray(
            [[0.4959, 0.5306, 0.0000], [0.1823, 0.5420, 0.1832],
             [0.0000, 0.0000, 0.4404]],
            dtype=np.float32)
        np_expect_ious_aligned = np.asarray([0.4959, 0.5420, 0.4404],
                                            dtype=np.float32)

        boxes1 = torch.from_numpy(np_boxes1)
        boxes2 = torch.from_numpy(np_boxes2)

        # test cw angle definition
        ious = box_iou_rotated(boxes1, boxes2, mode='iof')
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)
        ious = box_iou_rotated(boxes1, boxes2, mode='iof', aligned=True)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-4)

        # test ccw angle definition
        boxes1[..., -1] *= -1
        boxes2[..., -1] *= -1
        ious = box_iou_rotated(boxes1, boxes2, mode='iof', clockwise=False)
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)
        ious = box_iou_rotated(
            boxes1, boxes2, mode='iof', aligned=True, clockwise=False)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-4)

    @pytest.mark.parametrize('device', [
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
        pytest.param(
            'mlu',
            marks=pytest.mark.skipif(
                not IS_MLU_AVAILABLE, reason='requires MLU support')),
        pytest.param(
            'npu',
            marks=pytest.mark.skipif(
                not IS_NPU_AVAILABLE, reason='requires NPU support'))
    ])
    def test_box_iou_rotated_iof(self, device):
        np_boxes1 = np.asarray(
            [[1.0, 1.0, 3.0, 4.0, 0.5], [2.0, 2.0, 3.0, 4.0, 0.6],
             [7.0, 7.0, 8.0, 8.0, 0.4]],
            dtype=np.float32)
        np_boxes2 = np.asarray(
            [[0.0, 2.0, 2.0, 5.0, 0.3], [2.0, 1.0, 3.0, 3.0, 0.5],
             [5.0, 5.0, 6.0, 7.0, 0.4]],
            dtype=np.float32)
        np_expect_ious = np.asarray(
            [[0.4959, 0.5306, 0.0000], [0.1823, 0.5420, 0.1832],
             [0.0000, 0.0000, 0.4404]],
            dtype=np.float32)
        np_expect_ious_aligned = np.asarray([0.4959, 0.5420, 0.4404],
                                            dtype=np.float32)

        boxes1 = torch.from_numpy(np_boxes1).to(device)
        boxes2 = torch.from_numpy(np_boxes2).to(device)

        # test cw angle definition
        ious = box_iou_rotated(boxes1, boxes2, mode='iof')
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)

        ious = box_iou_rotated(boxes1, boxes2, mode='iof', aligned=True)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-4)

        # test ccw angle definition
        boxes1[..., -1] *= -1
        boxes2[..., -1] *= -1
        ious = box_iou_rotated(boxes1, boxes2, mode='iof', clockwise=False)
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)

        ious = box_iou_rotated(
            boxes1, boxes2, mode='iof', aligned=True, clockwise=False)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-4)
