import numpy as np
import pytest
import torch

from mmcv.ops import nms_rotated


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='GPU is required to test NMSRotated op')
class TestNmsRotated:

    def setup_class(self):
        self.np_boxes = np.array(
            [[6.0, 3.0, 8.0, 7.0, 0.5, 0.7], [3.0, 6.0, 9.0, 11.0, 0.6, 0.8],
             [3.0, 7.0, 10.0, 12.0, 0.3, 0.5], [1.0, 4.0, 13.0, 7.0, 0.6, 0.9]
             ],
            dtype=np.float32)
        self.np_labels = np.array([1, 0, 1, 0], dtype=np.float32)
        self.np_expect_dets = np.array(
            [[1.0, 4.0, 13.0, 7.0, 0.6], [3.0, 6.0, 9.0, 11.0, 0.6],
             [6.0, 3.0, 8.0, 7.0, 0.5]],
            dtype=np.float32)
        self.np_expect_keep_inds = np.array([3, 1, 0], dtype=np.int64)

    def test_ml_nms_rotated(self):
        boxes = torch.from_numpy(self.np_boxes).cuda()
        labels = torch.from_numpy(self.np_labels).cuda()
        dets, keep_inds = nms_rotated(boxes[:, :5], boxes[:, -1], 0.5, labels)
        assert np.allclose(dets.cpu().numpy()[:, :5], self.np_expect_dets)
        assert np.allclose(keep_inds.cpu().numpy(), self.np_expect_keep_inds)

    def test_nms_rotated(self):
        boxes = torch.from_numpy(self.np_boxes).cuda()
        dets, keep_inds = nms_rotated(boxes[:, :5], boxes[:, -1], 0.5)
        assert np.allclose(dets.cpu().numpy()[:, :5], self.np_expect_dets)
        assert np.allclose(keep_inds.cpu().numpy(), self.np_expect_keep_inds)
