# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.utils import IS_CUDA_AVAILABLE


class TestNMSQuadri:

    @pytest.mark.parametrize('device', [
        'cpu',
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
    ])
    def test_ml_nms_quadri(self, device):
        from mmcv.ops import nms_quadri
        np_boxes = np.array([[1.0, 1.0, 3.0, 4.0, 4.0, 4.0, 4.0, 1.0, 0.7],
                             [2.0, 2.0, 3.0, 4.0, 4.0, 2.0, 3.0, 1.0, 0.8],
                             [7.0, 7.0, 8.0, 8.0, 9.0, 7.0, 8.0, 6.0, 0.5],
                             [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.9]],
                            dtype=np.float32)
        np_labels = np.array([1, 0, 1, 0], dtype=np.float32)

        np_expect_dets = np.array([[0., 0., 0., 2., 2., 2., 2., 0.],
                                   [2., 2., 3., 4., 4., 2., 3., 1.],
                                   [7., 7., 8., 8., 9., 7., 8., 6.]],
                                  dtype=np.float32)
        np_expect_keep_inds = np.array([3, 1, 2], dtype=np.int64)

        boxes = torch.from_numpy(np_boxes).to(device)
        labels = torch.from_numpy(np_labels).to(device)

        dets, keep_inds = nms_quadri(boxes[:, :8], boxes[:, -1], 0.3, labels)

        assert np.allclose(dets.cpu().numpy()[:, :8], np_expect_dets)
        assert np.allclose(keep_inds.cpu().numpy(), np_expect_keep_inds)

    @pytest.mark.parametrize('device', [
        'cpu',
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
    ])
    def test_nms_quadri(self, device):
        from mmcv.ops import nms_quadri
        np_boxes = np.array([[1.0, 1.0, 3.0, 4.0, 4.0, 4.0, 4.0, 1.0, 0.7],
                             [2.0, 2.0, 3.0, 4.0, 4.0, 2.0, 3.0, 1.0, 0.8],
                             [7.0, 7.0, 8.0, 8.0, 9.0, 7.0, 8.0, 6.0, 0.5],
                             [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.9]],
                            dtype=np.float32)

        np_expect_dets = np.array([[0., 0., 0., 2., 2., 2., 2., 0.],
                                   [2., 2., 3., 4., 4., 2., 3., 1.],
                                   [7., 7., 8., 8., 9., 7., 8., 6.]],
                                  dtype=np.float32)
        np_expect_keep_inds = np.array([3, 1, 2], dtype=np.int64)

        boxes = torch.from_numpy(np_boxes).to(device)

        dets, keep_inds = nms_quadri(boxes[:, :8], boxes[:, -1], 0.3)
        assert np.allclose(dets.cpu().numpy()[:, :8], np_expect_dets)
        assert np.allclose(keep_inds.cpu().numpy(), np_expect_keep_inds)

    @pytest.mark.parametrize('device', [
        'cpu',
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
    ])
    def test_batched_nms(self, device):
        # test batched_nms with nms_quadri
        from mmcv.ops import batched_nms

        np_boxes = np.array([[1.0, 1.0, 3.0, 4.0, 4.0, 4.0, 4.0, 1.0, 0.7],
                             [2.0, 2.0, 3.0, 4.0, 4.0, 2.0, 3.0, 1.0, 0.8],
                             [7.0, 7.0, 8.0, 8.0, 9.0, 7.0, 8.0, 6.0, 0.5],
                             [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.9]],
                            dtype=np.float32)
        np_labels = np.array([1, 0, 1, 0], dtype=np.float32)

        np_expect_agnostic_dets = np.array([[0., 0., 0., 2., 2., 2., 2., 0.],
                                            [2., 2., 3., 4., 4., 2., 3., 1.],
                                            [7., 7., 8., 8., 9., 7., 8., 6.]],
                                           dtype=np.float32)
        np_expect_agnostic_keep_inds = np.array([3, 1, 2], dtype=np.int64)

        np_expect_dets = np.array([[0., 0., 0., 2., 2., 2., 2., 0.],
                                   [2., 2., 3., 4., 4., 2., 3., 1.],
                                   [1., 1., 3., 4., 4., 4., 4., 1.],
                                   [7., 7., 8., 8., 9., 7., 8., 6.]],
                                  dtype=np.float32)
        np_expect_keep_inds = np.array([3, 1, 0, 2], dtype=np.int64)

        nms_cfg = dict(type='nms_quadri', iou_threshold=0.3)

        # test class_agnostic is True
        boxes, keep = batched_nms(
            torch.from_numpy(np_boxes[:, :8]).to(device),
            torch.from_numpy(np_boxes[:, -1]).to(device),
            torch.from_numpy(np_labels).to(device),
            nms_cfg,
            class_agnostic=True)
        assert np.allclose(boxes.cpu().numpy()[:, :8], np_expect_agnostic_dets)
        assert np.allclose(keep.cpu().numpy(), np_expect_agnostic_keep_inds)

        # test class_agnostic is False
        boxes, keep = batched_nms(
            torch.from_numpy(np_boxes[:, :8]).to(device),
            torch.from_numpy(np_boxes[:, -1]).to(device),
            torch.from_numpy(np_labels).to(device),
            nms_cfg,
            class_agnostic=False)
        assert np.allclose(boxes.cpu().numpy()[:, :8], np_expect_dets)
        assert np.allclose(keep.cpu().numpy(), np_expect_keep_inds)
