# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.ops import gather_points
from mmcv.utils import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE


class TestGatherPoints:

    @pytest.mark.parametrize('device', [
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
        pytest.param(
            'npu',
            marks=pytest.mark.skipif(
                not IS_NPU_AVAILABLE, reason='requires NPU support'))
    ])
    def test_gather_points_all_close(self, device):
        features = torch.tensor(
            [[[
                -1.6095, -0.1029, -0.8876, -1.2447, -2.4031, 0.3708, -1.1586,
                -1.4967, -0.4800, 0.2252
            ],
              [
                  1.9138, 3.4979, 1.6854, 1.5631, 3.6776, 3.1154, 2.1705,
                  2.5221, 2.0411, 3.1446
              ],
              [
                  -1.4173, 0.3073, -1.4339, -1.4340, -1.2770, -0.2867, -1.4162,
                  -1.4044, -1.4245, -1.4074
              ]],
             [[
                 0.2160, 0.0842, 0.3661, -0.2749, -0.4909, -0.6066, -0.8773,
                 -0.0745, -0.9496, 0.1434
             ],
              [
                  1.3644, 1.8087, 1.6855, 1.9563, 1.2746, 1.9662, 0.9566,
                  1.8778, 1.1437, 1.3639
              ],
              [
                  -0.7172, 0.1692, 0.2241, 0.0721, -0.7540, 0.0462, -0.6227,
                  0.3223, -0.6944, -0.5294
              ]]],
            dtype=torch.float,
            device=device)
        idx = torch.tensor([[0, 1, 4, 0, 0, 0], [0, 5, 6, 0, 0, 0]],
                           dtype=torch.int32,
                           device=device)
        output = gather_points(features, idx)
        expected_output = torch.tensor(
            [[[-1.6095, -0.1029, -2.4031, -1.6095, -1.6095, -1.6095],
              [1.9138, 3.4979, 3.6776, 1.9138, 1.9138, 1.9138],
              [-1.4173, 0.3073, -1.2770, -1.4173, -1.4173, -1.4173]],
             [[0.2160, -0.6066, -0.8773, 0.2160, 0.2160, 0.2160],
              [1.3644, 1.9662, 0.9566, 1.3644, 1.3644, 1.3644],
              [-0.7172, 0.0462, -0.6227, -0.7172, -0.7172, -0.7172]]],
            dtype=torch.float,
            device=device)

        assert torch.allclose(output, expected_output)

        # test fp16
        output_half = gather_points(features.half(), idx)
        assert torch.allclose(output_half, expected_output.half())
