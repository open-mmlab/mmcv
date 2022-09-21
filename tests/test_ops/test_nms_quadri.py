# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='GPU is required to test NMSQuadri op')
class TestNmsQuadri:

    def test_ml_nms_quadri(self):
        from mmcv.ops import nms_quadri
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

        boxes = torch.from_numpy(np_boxes).cuda()
        labels = torch.from_numpy(np_labels).cuda()

        dets, keep_inds = nms_quadri(boxes[:, :8], boxes[:, -1], 0.5, labels)

        assert np.allclose(dets.cpu().numpy()[:, :8], np_expect_dets)
        assert np.allclose(keep_inds.cpu().numpy(), np_expect_keep_inds)

    def test_nms_quadri(self):
        from mmcv.ops import nms_quadri
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

        boxes = torch.from_numpy(np_boxes).cuda()

        dets, keep_inds = nms_quadri(boxes[:, :8], boxes[:, -1], 0.5)
        assert np.allclose(dets.cpu().numpy()[:, :8], np_expect_dets)
        assert np.allclose(keep_inds.cpu().numpy(), np_expect_keep_inds)

    def test_batched_nms(self):
        # test batched_nms with nms_quadri
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

        nms_cfg = dict(type='nms_quadri', iou_threshold=0.5)

        # test class_agnostic is True
        boxes, keep = batched_nms(
            torch.from_numpy(np_boxes[:, :8]),
            torch.from_numpy(np_boxes[:, -1]),
            torch.from_numpy(np_labels),
            nms_cfg,
            class_agnostic=True)
        assert np.allclose(boxes.cpu().numpy()[:, :8], np_expect_agnostic_dets)
        assert np.allclose(keep.cpu().numpy(), np_expect_agnostic_keep_inds)

        # test class_agnostic is False
        boxes, keep = batched_nms(
            torch.from_numpy(np_boxes[:, :8]),
            torch.from_numpy(np_boxes[:, -1]),
            torch.from_numpy(np_labels),
            nms_cfg,
            class_agnostic=False)
        assert np.allclose(boxes.cpu().numpy()[:, :8], np_expect_dets)
        assert np.allclose(keep.cpu().numpy(), np_expect_keep_inds)
