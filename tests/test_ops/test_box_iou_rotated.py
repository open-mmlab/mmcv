import numpy as np
import pytest
import torch

from mmcv.ops import box_iou_rotated


class TestBoxIoURotated(object):

    def setup_class(self):
        self.np_boxes1 = np.asarray(
            [[1.0, 1.0, 3.0, 4.0, 0.5], [2.0, 2.0, 3.0, 4.0, 0.6],
             [7.0, 7.0, 8.0, 8.0, 0.4]],
            dtype=np.float32)
        self.np_boxes2 = np.asarray(
            [[0.0, 2.0, 2.0, 5.0, 0.3], [2.0, 1.0, 3.0, 3.0, 0.5],
             [5.0, 5.0, 6.0, 7.0, 0.4]],
            dtype=np.float32)
        self.np_expect_ious_iou = np.asarray(
            [[0.3708, 0.4351, 0.0000], [0.1104, 0.4487, 0.0424],
             [0.0000, 0.0000, 0.3622]],
            dtype=np.float32)
        self.np_expect_ious_aligned_iou = np.asarray([0.3708, 0.4487, 0.3622],
                                                     dtype=np.float32)
        self.np_expect_ious_iof = np.asarray(
            [[0.4959, 0.5306, 0.0000], [0.1823, 0.5420, 0.1832],
             [0.0000, 0.0000, 0.4404]],
            dtype=np.float32)
        self.np_expect_ious_aligned_iof = np.asarray([0.4959, 0.5420, 0.4404],
                                                     dtype=np.float32)
        self.np_expect_ious = {
            'iou': {
                False: self.np_expect_ious_iou,
                True: self.np_expect_ious_aligned_iou
            },
            'iof': {
                False: self.np_expect_ious_iof,
                True: self.np_expect_ious_aligned_iof
            }
        }

    @pytest.mark.parametrize('device', [
        'cpu',
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason='requires CUDA support'))
    ])
    @pytest.mark.parametrize('mode', ['iou', 'iof'])
    @pytest.mark.parametrize('aligned', [False, True])
    def test_box_iou_rotated(self, device, mode, aligned):
        boxes1 = torch.from_numpy(self.np_boxes1)
        boxes2 = torch.from_numpy(self.np_boxes2)
        if device == 'cuda':
            boxes1 = boxes1.cuda()
            boxes2 = boxes2.cuda()

        ious = box_iou_rotated(boxes1, boxes2, mode=mode, aligned=aligned)
        assert np.allclose(
            ious.cpu().numpy(), self.np_expect_ious[mode][aligned], atol=1e-4)
