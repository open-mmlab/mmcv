# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.utils import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE

from torch.autograd import gradcheck, gradgradcheck


class TestFusedBiasLeakyReLU:

    @classmethod
    def setup_class(cls):
        if not IS_CUDA_AVAILABLE and not IS_NPU_AVAILABLE:
            return
        if IS_CUDA_AVAILABLE:
            cls.input_tensor = torch.randn((2, 2, 2, 2),
                                           requires_grad=True).cuda()
            cls.bias = torch.zeros(2, requires_grad=True).cuda()
        elif IS_NPU_AVAILABLE:
            cls.input_tensor = torch.randn((2, 2, 2, 2),
                                           requires_grad=True).npu()
            cls.bias = torch.zeros(2, requires_grad=True).npu()

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
    def test_gradient(self, device):

        from mmcv.ops import FusedBiasLeakyReLU
        gradcheck(
            FusedBiasLeakyReLU(2).to(device),
            self.input_tensor,
            eps=1e-4,
            atol=1e-3)

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
    def test_gradgradient(self, device):

        from mmcv.ops import FusedBiasLeakyReLU
        gradgradcheck(
            FusedBiasLeakyReLU(2).to(device),
            self.input_tensor,
            eps=1e-4,
            atol=1e-3)
