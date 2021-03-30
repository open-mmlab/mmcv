import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck


class TestFusedBiasLeakyReLU(object):

    @classmethod
    def setup_class(cls):
        if not torch.cuda.is_available():
            return
        cls.input_tensor = torch.randn((2, 2, 2, 2), requires_grad=True).cuda()
        cls.bias = torch.zeros(2, requires_grad=True).cuda()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_gradient(self):

        from mmcv.ops import FusedBiasLeakyReLU
        gradcheck(
            FusedBiasLeakyReLU(2).cuda(),
            self.input_tensor,
            eps=1e-4,
            atol=1e-3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_gradgradient(self):

        from mmcv.ops import FusedBiasLeakyReLU
        gradgradcheck(
            FusedBiasLeakyReLU(2).cuda(),
            self.input_tensor,
            eps=1e-4,
            atol=1e-3)
