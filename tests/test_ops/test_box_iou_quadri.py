# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch


class TestBoxIoUQuadri:

    def test_box_iou_quadri_cpu(self):
        from mmcv.ops import box_iou_quadri
        np_boxes1 = np.asarray([[1.0, 1.0, 3.0, 4.0, 4.0, 4.0, 4.0, 1.0],
                                [2.0, 2.0, 3.0, 4.0, 4.0, 2.0, 3.0, 1.0],
                                [7.0, 7.0, 8.0, 8.0, 9.0, 7.0, 8.0, 6.0]],
                               dtype=np.float32)
        np_boxes2 = np.asarray([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0],
                                [2.0, 1.0, 2.0, 4.0, 4.0, 4.0, 4.0, 1.0],
                                [7.0, 6.0, 7.0, 8.0, 9.0, 8.0, 9.0, 6.0]],
                               dtype=np.float32)
        np_expect_ious = np.asarray(
            [[0.0714, 1.0000, 0.0000], [0.0000, 0.5000, 0.0000],
             [0.0000, 0.0000, 0.5000]],
            dtype=np.float32)
        np_expect_ious_aligned = np.asarray([0.0714, 0.5000, 0.5000],
                                            dtype=np.float32)

        boxes1 = torch.from_numpy(np_boxes1)
        boxes2 = torch.from_numpy(np_boxes2)

        ious = box_iou_quadri(boxes1, boxes2)
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)

        ious = box_iou_quadri(boxes1, boxes2, aligned=True)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-4)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires CUDA support')
    def test_box_iou_quadri_cuda(self):
        from mmcv.ops import box_iou_quadri
        np_boxes1 = np.asarray([[1.0, 1.0, 3.0, 4.0, 4.0, 4.0, 4.0, 1.0],
                                [2.0, 2.0, 3.0, 4.0, 4.0, 2.0, 3.0, 1.0],
                                [7.0, 7.0, 8.0, 8.0, 9.0, 7.0, 8.0, 6.0]],
                               dtype=np.float32)
        np_boxes2 = np.asarray([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0],
                                [2.0, 1.0, 2.0, 4.0, 4.0, 4.0, 4.0, 1.0],
                                [7.0, 6.0, 7.0, 8.0, 9.0, 8.0, 9.0, 6.0]],
                               dtype=np.float32)
        np_expect_ious = np.asarray(
            [[0.0714, 1.2222, 0.0000], [0.1570, 0.5000, 0.0000],
             [0.0000, 0.0000, 0.5000]],
            dtype=np.float32)
        np_expect_ious_aligned = np.asarray([0.0714, 0.5000, 0.5000],
                                            dtype=np.float32)

        boxes1 = torch.from_numpy(np_boxes1).cuda()
        boxes2 = torch.from_numpy(np_boxes2).cuda()

        ious = box_iou_quadri(boxes1, boxes2)
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)

        ious = box_iou_quadri(boxes1, boxes2, aligned=True)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-4)

    def test_box_iou_quadri_iof_cpu(self):
        from mmcv.ops import box_iou_quadri
        np_boxes1 = np.asarray([[1.0, 1.0, 3.0, 4.0, 4.0, 4.0, 4.0, 1.0],
                                [2.0, 2.0, 3.0, 4.0, 4.0, 2.0, 3.0, 1.0],
                                [7.0, 7.0, 8.0, 8.0, 9.0, 7.0, 8.0, 6.0]],
                               dtype=np.float32)
        np_boxes2 = np.asarray([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0],
                                [2.0, 1.0, 2.0, 4.0, 4.0, 4.0, 4.0, 1.0],
                                [7.0, 6.0, 7.0, 8.0, 9.0, 8.0, 9.0, 6.0]],
                               dtype=np.float32)
        np_expect_ious = np.asarray(
            [[0.1111, 1.0000, 0.0000], [0.0000, 1.0000, 0.0000],
             [0.0000, 0.0000, 1.0000]],
            dtype=np.float32)
        np_expect_ious_aligned = np.asarray([0.1111, 1.0000, 1.0000],
                                            dtype=np.float32)

        boxes1 = torch.from_numpy(np_boxes1)
        boxes2 = torch.from_numpy(np_boxes2)

        ious = box_iou_quadri(boxes1, boxes2, mode='iof')
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)
        ious = box_iou_quadri(boxes1, boxes2, mode='iof', aligned=True)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-4)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires CUDA support')
    def test_box_iou_quadri_iof_cuda(self):
        from mmcv.ops import box_iou_quadri
        np_boxes1 = np.asarray([[1.0, 1.0, 3.0, 4.0, 4.0, 4.0, 4.0, 1.0],
                                [2.0, 2.0, 3.0, 4.0, 4.0, 2.0, 3.0, 1.0],
                                [7.0, 7.0, 8.0, 8.0, 9.0, 7.0, 8.0, 6.0]],
                               dtype=np.float32)
        np_boxes2 = np.asarray([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0],
                                [2.0, 1.0, 2.0, 4.0, 4.0, 4.0, 4.0, 1.0],
                                [7.0, 6.0, 7.0, 8.0, 9.0, 8.0, 9.0, 6.0]],
                               dtype=np.float32)
        np_expect_ious = np.asarray(
            [[0.1111, 0.9167, 0.0000], [0.5429, 1.0000, 0.0000],
             [0.0000, 0.0000, 1.0000]],
            dtype=np.float32)
        np_expect_ious_aligned = np.asarray([0.1111, 1.0000, 1.0000],
                                            dtype=np.float32)

        boxes1 = torch.from_numpy(np_boxes1).cuda()
        boxes2 = torch.from_numpy(np_boxes2).cuda()

        ious = box_iou_quadri(boxes1, boxes2, mode='iof')
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)

        ious = box_iou_quadri(boxes1, boxes2, mode='iof', aligned=True)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-4)
