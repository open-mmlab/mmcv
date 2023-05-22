# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE, IS_NPU_AVAILABLE


class TestMaskedConv2d:

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
    def test_masked_conv2d_all_close(self, device):
        from mmcv.ops import MaskedConv2d
        np_input = np.load(
            'tests/data/for_masked_conv2d/masked_conv2d_for_input.npy')
        np_mask = np.load(
            'tests/data/for_masked_conv2d/masked_conv2d_for_mask.npy')
        np_weight = np.load(
            'tests/data/for_masked_conv2d/masked_conv2d_for_weight.npy')
        np_bias = np.load(
            'tests/data/for_masked_conv2d/masked_conv2d_for_bias.npy')
        np_output = np.load(
            'tests/data/for_masked_conv2d/masked_conv2d_for_output.npy')
        input = torch.tensor(np_input, dtype=torch.float, device=device)
        mask = torch.tensor(np_mask, dtype=torch.float, device=device)
        weight = torch.tensor(np_weight, dtype=torch.float, device=device)
        bias = torch.tensor(np_bias, dtype=torch.float, device=device)
        conv = MaskedConv2d(3, 3, 3, 1, 1).to(device)
        conv.weight = torch.nn.Parameter(weight)
        conv.bias = torch.nn.Parameter(bias)
        output = conv(input, mask)
        assert np.allclose(output.data.cpu().numpy(), np_output, 1e-3)
