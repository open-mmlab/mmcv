# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.utils import IS_CUDA_AVAILABLE


class TestBoxIoUQuadri:

    @pytest.mark.parametrize('device', [
        'cpu',
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
    ])
    def test_box_iou_quadri_cuda(self, device):
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

        boxes1 = torch.from_numpy(np_boxes1).to(device)
        boxes2 = torch.from_numpy(np_boxes2).to(device)

        ious = box_iou_quadri(boxes1, boxes2)
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)

        ious = box_iou_quadri(boxes1, boxes2, aligned=True)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-4)

    @pytest.mark.parametrize('device', [
        'cpu',
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
    ])
    def test_box_iou_quadri_iof_cuda(self, device):
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

        boxes1 = torch.from_numpy(np_boxes1).to(device)
        boxes2 = torch.from_numpy(np_boxes2).to(device)

        ious = box_iou_quadri(boxes1, boxes2, mode='iof')
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)

        ious = box_iou_quadri(boxes1, boxes2, mode='iof', aligned=True)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-4)
