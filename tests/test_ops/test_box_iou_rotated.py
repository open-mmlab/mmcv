# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

try:
    # If PyTorch version >= 1.6.0 and fp16 is enabled, torch.cuda.amp.autocast
    # would be imported and used; we should test if our modules support it.
    from torch.cuda.amp import autocast
    _IS_AUTOCAST_AVAILABLE = True
except ImportError:
    _IS_AUTOCAST_AVAILABLE = False


class TestBoxIoURotated:

    def test_box_iou_rotated_cpu(self):
        from mmcv.ops import box_iou_rotated
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

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires CUDA support')
    def test_box_iou_rotated_cuda(self):
        from mmcv.ops import box_iou_rotated
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

        boxes1 = torch.from_numpy(np_boxes1).cuda()
        boxes2 = torch.from_numpy(np_boxes2).cuda()

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

    @pytest.mark.skipif(
        not _IS_AUTOCAST_AVAILABLE, reason='requires autocast support')
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires CUDA support')
    def test_box_iou_rotated_with_autocast(self, device):
        from mmcv.ops import box_iou_rotated
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
            dtype=np.float16)

        boxes1 = torch.from_numpy(np_boxes1).to(device).type(torch.half)
        boxes2 = torch.from_numpy(np_boxes2).to(device)

        with autocast(enabled=True):
            # test cw angle definition
            ious = box_iou_rotated(boxes1, boxes2)
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-2)

    def test_box_iou_rotated_iof_cpu(self):
        from mmcv.ops import box_iou_rotated
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

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires CUDA support')
    def test_box_iou_rotated_iof_cuda(self):
        from mmcv.ops import box_iou_rotated
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

        boxes1 = torch.from_numpy(np_boxes1).cuda()
        boxes2 = torch.from_numpy(np_boxes2).cuda()

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
