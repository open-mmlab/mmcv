import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class TestBilinearGridSample(object):

    def _test_bilinear_grid_sample(self,
                                   dtype=torch.float,
                                   align_corners=False,
                                   precision=1e-3):
        from mmcv.ops.point_sample import bilinear_grid_sample

        input = torch.rand(1, 1, 10, 10, dtype=dtype)
        grid = torch.Tensor([[[1, 0, 0], [0, 1, 0]]])
        grid = nn.functional.affine_grid(grid, (1, 1, 15, 15)).type_as(input)

        out = bilinear_grid_sample(input, grid, align_corners=align_corners)
        ref_out = F.grid_sample(input, grid, align_corners=align_corners)

        assert np.allclose(out.data.detach().cpu().numpy(),
                           ref_out.data.detach().cpu().numpy(), precision)

    def test_bilinear_grid_sample(self):
        self._test_bilinear_grid_sample(torch.double, False)
        self._test_bilinear_grid_sample(torch.double, True)
        self._test_bilinear_grid_sample(torch.float, False)
        self._test_bilinear_grid_sample(torch.float, True)
