import numpy as np
import pytest
import torch

from mmcv.ops import bbox_overlaps


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
class TestBBox(object):

    def setup_class(self):
        self.b1_1 = torch.tensor([[1.0, 1.0, 3.0, 4.0], [2.0, 2.0, 3.0, 4.0],
                                  [7.0, 7.0, 8.0, 8.0]]).cuda()
        self.b2_1 = torch.tensor([[0.0, 2.0, 2.0, 5.0], [2.0, 1.0, 3.0,
                                                         3.0]]).cuda()
        self.b1_2 = torch.tensor([[1.0, 1.0, 3.0, 4.0], [2.0, 2.0, 3.0,
                                                         4.0]]).cuda()
        self.b2_2 = torch.tensor([[0.0, 2.0, 2.0, 5.0], [2.0, 1.0, 3.0,
                                                         3.0]]).cuda()
        self.b1_3 = torch.tensor([[0.0, 0.0, 3.0, 3.0]]).cuda()
        self.b2_3 = torch.tensor([[4.0, 0.0, 5.0, 3.0], [3.0, 0.0, 4.0, 3.0],
                                  [2.0, 0.0, 3.0, 3.0], [1.0, 0.0, 2.0,
                                                         3.0]]).cuda()

        self.should_output_1 = np.array([[0.33333334, 0.5], [0.2, 0.5],
                                         [0.0, 0.0]])
        self.should_output_2 = np.array([0.33333334, 0.5])
        self.should_output_3 = np.array([0, 0.2, 0.5, 0.5])

        self.b1s = [self.b1_1, self.b1_2, self.b1_3]
        self.b2s = [self.b2_1, self.b2_2, self.b2_3]
        self.aligneds = [False, True, False]
        self.should_outputs = [
            self.should_output_1, self.should_output_2, self.should_output_3
        ]

    def _test_bbox_overlaps(self, dtype=torch.float):
        for b1, b2, aligned, should_output in zip(self.b1s, self.b2s,
                                                  self.aligneds,
                                                  self.should_outputs):
            b1 = b1.type(dtype)
            b2 = b2.type(dtype)
            out = bbox_overlaps(
                bboxes1=b1, bboxes2=b2, aligned=aligned, offset=1)
            assert np.allclose(out.cpu().numpy(), should_output, 1e-2)

    @pytest.mark.parametrize('dtype', [torch.float, torch.half])
    def test_bbox_overlaps_float(self, dtype):
        self._test_bbox_overlaps(dtype)
