import numpy as np
import pytest
import torch

from mmcv.ops import bbox_overlaps


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
class TestBBox(object):

    def setup_class(self):
        self.bbox_1_1 = torch.tensor([[1.0, 1.0, 3.0,
                                       4.0], [2.0, 2.0, 3.0, 4.0],
                                      [7.0, 7.0, 8.0, 8.0]]).cuda()
        self.bbox_2_1 = torch.tensor([[0.0, 2.0, 2.0, 5.0],
                                      [2.0, 1.0, 3.0, 3.0]]).cuda()
        self.bbox_1_2 = torch.tensor([[1.0, 1.0, 3.0, 4.0],
                                      [2.0, 2.0, 3.0, 4.0]]).cuda()
        self.bbox_2_2 = torch.tensor([[0.0, 2.0, 2.0, 5.0],
                                      [2.0, 1.0, 3.0, 3.0]]).cuda()
        self.bbox_1_3 = torch.tensor([[0.0, 0.0, 3.0, 3.0]]).cuda()
        self.bbox_2_3 = torch.tensor([[4.0, 0.0, 5.0,
                                       3.0], [3.0, 0.0, 4.0, 3.0],
                                      [2.0, 0.0, 3.0, 3.0],
                                      [1.0, 0.0, 2.0, 3.0]]).cuda()

        self.expected_output_1 = np.array([[0.33333334, 0.5], [0.2, 0.5],
                                           [0.0, 0.0]])
        self.expected_output_2 = np.array([0.33333334, 0.5])
        self.expected_output_3 = np.array([0, 0.2, 0.5, 0.5])

        self.bboxes_1 = [self.bbox_1_1, self.bbox_1_2, self.bbox_1_3]
        self.bboxes_2 = [self.bbox_2_1, self.bbox_2_2, self.bbox_2_3]
        self.aligned_flags = [False, True, False]
        self.expected_outputs = [
            self.expected_output_1, self.expected_output_2,
            self.expected_output_3
        ]

    def _test_bbox_overlaps(self, dtype=torch.float):
        for b1, b2, aligned, expected_output in zip(self.bboxes_1,
                                                    self.bboxes_2,
                                                    self.aligned_flags,
                                                    self.expected_outputs):
            b1 = b1.type(dtype)
            b2 = b2.type(dtype)
            out = bbox_overlaps(
                bboxes1=b1, bboxes2=b2, aligned=aligned, offset=1)
            assert np.allclose(out.cpu().numpy(), expected_output, 1e-2)

    @pytest.mark.parametrize('dtype', [torch.float, torch.half])
    def test_bbox_overlaps_float(self, dtype):
        self._test_bbox_overlaps(dtype)
