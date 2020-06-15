import numpy as np
import torch


class TestBBox(object):

    def _test_bbox_overlaps(self, dtype=torch.float):

        if not torch.cuda.is_available():
            return
        from mmcv.op import bbox_overlaps
        b1 = torch.tensor([[1.0, 1.0, 3.0, 4.0], [2.0, 2.0, 3.0, 4.0],
                           [7.0, 7.0, 8.0, 8.0]]).cuda().type(dtype)
        b2 = torch.tensor([[0.0, 2.0, 2.0, 5.0], [2.0, 1.0, 3.0,
                                                  3.0]]).cuda().type(dtype)
        should_output = np.array([[0.33333334, 0.5], [0.2, 0.5], [0.0, 0.0]])
        out = bbox_overlaps(b1, b2, offset=1)
        assert np.allclose(out.cpu().numpy(), should_output, 1e-2)

        b1 = torch.tensor([[1.0, 1.0, 3.0, 4.0], [2.0, 2.0, 3.0,
                                                  4.0]]).cuda().type(dtype)
        b2 = torch.tensor([[0.0, 2.0, 2.0, 5.0], [2.0, 1.0, 3.0,
                                                  3.0]]).cuda().type(dtype)
        should_output = np.array([0.33333334, 0.5])
        out = bbox_overlaps(b1, b2, aligned=True, offset=1)
        assert np.allclose(out.cpu().numpy(), should_output, 1e-2)

        b1 = torch.tensor([[0.0, 0.0, 3.0, 3.0]]).cuda().type(dtype)
        b1 = torch.tensor([[0.0, 0.0, 3.0, 3.0]]).cuda().type(dtype)
        b2 = torch.tensor([[4.0, 0.0, 5.0, 3.0], [3.0, 0.0, 4.0, 3.0],
                           [2.0, 0.0, 3.0, 3.0], [1.0, 0.0, 2.0,
                                                  3.0]]).cuda().type(dtype)
        should_output = np.array([0, 0.2, 0.5, 0.5])
        out = bbox_overlaps(b1, b2, offset=1)
        assert np.allclose(out.cpu().numpy(), should_output, 1e-2)

    def test_bbox_overlaps_float(self):
        self._test_bbox_overlaps(torch.float)

    def test_bbox_overlaps_half(self):
        self._test_bbox_overlaps(torch.half)
