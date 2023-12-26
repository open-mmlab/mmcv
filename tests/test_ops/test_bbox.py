# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.utils import (IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE, IS_MPS_AVAILABLE,
                        IS_NPU_AVAILABLE, digit_version)


class TestBBox:

    def _test_bbox_overlaps(self, device='cpu', dtype=torch.float):
        from mmcv.ops import bbox_overlaps
        b1 = torch.tensor([[1.0, 1.0, 3.0, 4.0], [2.0, 2.0, 3.0, 4.0],
                           [7.0, 7.0, 8.0, 8.0]]).to(device).type(dtype)
        b2 = torch.tensor([[0.0, 2.0, 2.0, 5.0], [2.0, 1.0, 3.0,
                                                  3.0]]).to(device).type(dtype)
        should_output = np.array([[0.33333334, 0.5], [0.2, 0.5], [0.0, 0.0]])
        out = bbox_overlaps(b1, b2, offset=1)
        assert np.allclose(out.cpu().numpy(), should_output, 1e-2)

        b1 = torch.tensor([[1.0, 1.0, 3.0, 4.0], [2.0, 2.0, 3.0,
                                                  4.0]]).to(device).type(dtype)
        b2 = torch.tensor([[0.0, 2.0, 2.0, 5.0], [2.0, 1.0, 3.0,
                                                  3.0]]).to(device).type(dtype)
        should_output = np.array([0.33333334, 0.5])
        out = bbox_overlaps(b1, b2, aligned=True, offset=1)
        assert np.allclose(out.cpu().numpy(), should_output, 1e-2)

        b1 = torch.tensor([[0.0, 0.0, 3.0, 3.0]]).to(device).type(dtype)
        b2 = torch.tensor([[4.0, 0.0, 5.0, 3.0], [3.0, 0.0, 4.0, 3.0],
                           [2.0, 0.0, 3.0, 3.0], [1.0, 0.0, 2.0,
                                                  3.0]]).to(device).type(dtype)
        should_output = np.array([0, 0.2, 0.5, 0.5])
        out = bbox_overlaps(b1, b2, offset=1)
        assert np.allclose(out.cpu().numpy(), should_output, 1e-2)

        b1 = torch.tensor([[10.0 + i, 10.0 + i, 30.0 + i, 30.0 + i]
                           for i in range(1000)]).to(device).type(dtype)
        b2 = torch.tensor([[20.0 + i, 20.0 + i, 40.0 + i, 40.0 + i]
                           for i in range(1000)]).to(device).type(dtype)
        should_output = np.array([1 / 7] * 1000)
        out = bbox_overlaps(b1, b2, aligned=True)
        assert np.allclose(out.cpu().numpy(), should_output, 1e-2)

    @pytest.mark.parametrize('device', [
        'cpu',
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
        pytest.param(
            'mlu',
            marks=pytest.mark.skipif(
                not IS_MLU_AVAILABLE, reason='requires MLU support')),
        pytest.param(
            'mps',
            marks=pytest.mark.skipif(
                not IS_MPS_AVAILABLE
                or digit_version(torch.__version__) >= digit_version('2.1.0'),
                reason='requires MPS support')),
        pytest.param(
            'npu',
            marks=pytest.mark.skipif(
                not IS_NPU_AVAILABLE, reason='requires NPU support'))
    ])
    def test_bbox_overlaps_float(self, device):
        self._test_bbox_overlaps(device, dtype=torch.float)

    @pytest.mark.parametrize('device', [
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
        pytest.param(
            'mlu',
            marks=pytest.mark.skipif(
                not IS_MLU_AVAILABLE, reason='requires MLU support')),
        pytest.param(
            'npu',
            marks=pytest.mark.skipif(
                not IS_NPU_AVAILABLE, reason='requires NPU support'))
    ])
    def test_bbox_overlaps_half(self, device):
        self._test_bbox_overlaps(device, dtype=torch.half)
