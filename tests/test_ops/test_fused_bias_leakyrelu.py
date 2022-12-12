# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.utils import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE

_USING_PARROTS = True
try:
    from parrots.autograd import gradcheck
except ImportError:
    from torch.autograd import gradcheck, gradgradcheck
    _USING_PARROTS = False


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
        if _USING_PARROTS:
            if IS_CUDA_AVAILABLE:
                gradcheck(
                    FusedBiasLeakyReLU(2).cuda(),
                    self.input_tensor,
                    delta=1e-4,
                    pt_atol=1e-3)
        else:
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
