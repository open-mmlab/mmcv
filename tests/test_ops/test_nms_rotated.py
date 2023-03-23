# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE, IS_NPU_AVAILABLE


class TestNmsRotated:

    @pytest.mark.parametrize('device', [
        pytest.param(
            'npu',
            marks=pytest.mark.skipif(
                not IS_NPU_AVAILABLE, reason='requires NPU support')),
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
        pytest.param(
            'mlu',
            marks=pytest.mark.skipif(
                not IS_MLU_AVAILABLE, reason='requires MLU support'))
    ])
    def test_ml_nms_rotated(self, device):
        from mmcv.ops import nms_rotated
        np_boxes = np.array(
            [[6.0, 3.0, 8.0, 7.0, 0.5, 0.7], [3.0, 6.0, 9.0, 11.0, 0.6, 0.8],
             [3.0, 7.0, 10.0, 12.0, 0.3, 0.5], [1.0, 4.0, 13.0, 7.0, 0.6, 0.9]
             ],
            dtype=np.float32)
        np_labels = np.array([1, 0, 1, 0], dtype=np.float32)

        np_expect_dets = np.array(
            [[1.0, 4.0, 13.0, 7.0, 0.6], [3.0, 6.0, 9.0, 11.0, 0.6],
             [6.0, 3.0, 8.0, 7.0, 0.5]],
            dtype=np.float32)
        np_expect_keep_inds = np.array([3, 1, 0], dtype=np.int64)

        boxes = torch.from_numpy(np_boxes).to(device)
        labels = torch.from_numpy(np_labels).to(device)

        # test cw angle definition
        dets, keep_inds = nms_rotated(boxes[:, :5], boxes[:, -1], 0.5, labels)

        assert np.allclose(dets.cpu().numpy()[:, :5], np_expect_dets)
        assert np.allclose(keep_inds.cpu().numpy(), np_expect_keep_inds)

        # test ccw angle definition
        boxes[..., -2] *= -1
        dets, keep_inds = nms_rotated(
            boxes[:, :5], boxes[:, -1], 0.5, labels, clockwise=False)
        dets[..., -2] *= -1
        assert np.allclose(dets.cpu().numpy()[:, :5], np_expect_dets)
        assert np.allclose(keep_inds.cpu().numpy(), np_expect_keep_inds)

    @pytest.mark.parametrize('device', [
        pytest.param(
            'npu',
            marks=pytest.mark.skipif(
                not IS_NPU_AVAILABLE, reason='requires NPU support')),
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
        pytest.param(
            'mlu',
            marks=pytest.mark.skipif(
                not IS_MLU_AVAILABLE, reason='requires MLU support'))
    ])
    def test_nms_rotated(self, device):
        from mmcv.ops import nms_rotated
        np_boxes = np.array(
            [[6.0, 3.0, 8.0, 7.0, 0.5, 0.7], [3.0, 6.0, 9.0, 11.0, 0.6, 0.8],
             [3.0, 7.0, 10.0, 12.0, 0.3, 0.5], [1.0, 4.0, 13.0, 7.0, 0.6, 0.9]
             ],
            dtype=np.float32)

        np_expect_dets = np.array(
            [[1.0, 4.0, 13.0, 7.0, 0.6], [3.0, 6.0, 9.0, 11.0, 0.6],
             [6.0, 3.0, 8.0, 7.0, 0.5]],
            dtype=np.float32)
        np_expect_keep_inds = np.array([3, 1, 0], dtype=np.int64)

        boxes = torch.from_numpy(np_boxes).to(device)

        # test cw angle definition
        dets, keep_inds = nms_rotated(boxes[:, :5], boxes[:, -1], 0.5)
        assert np.allclose(dets.cpu().numpy()[:, :5], np_expect_dets)
        assert np.allclose(keep_inds.cpu().numpy(), np_expect_keep_inds)

        # test ccw angle definition
        boxes[..., -2] *= -1
        dets, keep_inds = nms_rotated(
            boxes[:, :5], boxes[:, -1], 0.5, clockwise=False)
        dets[..., -2] *= -1
        assert np.allclose(dets.cpu().numpy()[:, :5], np_expect_dets)
        assert np.allclose(keep_inds.cpu().numpy(), np_expect_keep_inds)

    def test_batched_nms(self):
        # test batched_nms with nms_rotated
        from mmcv.ops import batched_nms

        np_boxes = np.array(
            [[6.0, 3.0, 8.0, 7.0, 0.5, 0.7], [3.0, 6.0, 9.0, 11.0, 0.6, 0.8],
             [3.0, 7.0, 10.0, 12.0, 0.3, 0.5], [1.0, 4.0, 13.0, 7.0, 0.6, 0.9]
             ],
            dtype=np.float32)
        np_labels = np.array([1, 0, 1, 0], dtype=np.float32)

        np_expect_agnostic_dets = np.array(
            [[1.0, 4.0, 13.0, 7.0, 0.6], [3.0, 6.0, 9.0, 11.0, 0.6],
             [6.0, 3.0, 8.0, 7.0, 0.5]],
            dtype=np.float32)
        np_expect_agnostic_keep_inds = np.array([3, 1, 0], dtype=np.int64)

        np_expect_dets = np.array(
            [[1.0, 4.0, 13.0, 7.0, 0.6], [3.0, 6.0, 9.0, 11.0, 0.6],
             [6.0, 3.0, 8.0, 7.0, 0.5], [3.0, 7.0, 10.0, 12.0, 0.3]],
            dtype=np.float32)
        np_expect_keep_inds = np.array([3, 1, 0, 2], dtype=np.int64)

        nms_cfg = dict(type='nms_rotated', iou_threshold=0.5)

        # test class_agnostic is True
        boxes, keep = batched_nms(
            torch.from_numpy(np_boxes[:, :5]),
            torch.from_numpy(np_boxes[:, -1]),
            torch.from_numpy(np_labels),
            nms_cfg,
            class_agnostic=True)
        assert np.allclose(boxes.cpu().numpy()[:, :5], np_expect_agnostic_dets)
        assert np.allclose(keep.cpu().numpy(), np_expect_agnostic_keep_inds)

        # test class_agnostic is False
        boxes, keep = batched_nms(
            torch.from_numpy(np_boxes[:, :5]),
            torch.from_numpy(np_boxes[:, -1]),
            torch.from_numpy(np_labels),
            nms_cfg,
            class_agnostic=False)
        assert np.allclose(boxes.cpu().numpy()[:, :5], np_expect_dets)
        assert np.allclose(keep.cpu().numpy(), np_expect_keep_inds)
