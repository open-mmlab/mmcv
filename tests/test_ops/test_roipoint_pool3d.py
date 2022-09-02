# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.ops import RoIPointPool3d
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE


class TestRoiPointPool3d:

    def _test_roipoint_pool3d_allclose(self, device, dtype):
        points = torch.tensor([[1, 2, 3.3], [1.2, 2.5, 3.0], [0.8, 2.1, 3.5],
                               [1.6, 2.6, 3.6], [0.8, 1.2, 3.9],
                               [-9.2, 21.0, 18.2], [3.8, 7.9, 6.3],
                               [4.7, 3.5, -12.2], [3.8, 7.6, -2],
                               [-10.6, -12.9, -20], [-16, -18, 9],
                               [-21.3, -52, -5], [0, 0, 0], [6, 7, 8],
                               [-2, -3,
                                -4]]).unsqueeze(0).to(device).type(dtype)
        feats = points.clone()
        rois = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.3],
                              [-10.0, 23.0, 16.0, 10, 20, 20,
                               0.5]]]).to(device).type(dtype)

        roipoint_pool3d = RoIPointPool3d(num_sampled_points=4)
        pooled_features, pooled_empty_flag = roipoint_pool3d(
            points, feats, rois)
        expected_pooled_features = torch.tensor(
            [[[[1, 2, 3.3, 1, 2, 3.3], [1.2, 2.5, 3, 1.2, 2.5, 3],
               [0.8, 2.1, 3.5, 0.8, 2.1, 3.5], [1.6, 2.6, 3.6, 1.6, 2.6, 3.6]],
              [[-9.2, 21, 18.2, -9.2, 21, 18.2],
               [-9.2, 21, 18.2, -9.2, 21, 18.2],
               [-9.2, 21, 18.2, -9.2, 21, 18.2],
               [-9.2, 21, 18.2, -9.2, 21, 18.2]]]]).to(device).type(dtype)
        expected_pooled_empty_flag = torch.tensor([[0, 0]]).int().to(device)

        assert torch.allclose(pooled_features, expected_pooled_features)
        assert torch.allclose(pooled_empty_flag, expected_pooled_empty_flag)

    @pytest.mark.parametrize('device', [
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
        pytest.param(
            'mlu',
            marks=pytest.mark.skipif(
                not IS_MLU_AVAILABLE, reason='requires MLU support'))
    ])
    @pytest.mark.parametrize('dtype', [
        torch.float, torch.half,
        pytest.param(
            torch.double,
            marks=pytest.mark.skipif(
                IS_MLU_AVAILABLE, reason='MLU does not support for double'))
    ])
    def test_roipoint_pool3d_allclose(self, device, dtype):
        self._test_roipoint_pool3d_allclose(device, dtype)
