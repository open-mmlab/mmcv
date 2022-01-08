import pytest
import torch

from mmcv.ops import FusedBiasLeakyReLU

_USING_PARROTS = True
try:
    from parrots.autograd import gradcheck
except ImportError:
    from torch.autograd import gradcheck, gradgradcheck
    _USING_PARROTS = False


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
class TestFusedBiasLeakyReLU(object):

    @classmethod
    def setup_class(cls):
        cls.input_tensor = torch.randn((2, 2, 2, 2), requires_grad=True).cuda()
        cls.bias = torch.zeros(2, requires_grad=True).cuda()

    def test_gradient(self):
        if _USING_PARROTS:
            gradcheck(
                FusedBiasLeakyReLU(2).cuda(),
                self.input_tensor,
                delta=1e-4,
                pt_atol=1e-3)
        else:
            gradcheck(
                FusedBiasLeakyReLU(2).cuda(),
                self.input_tensor,
                eps=1e-4,
                atol=1e-3)

    @pytest.mark.skipif(_USING_PARROTS, reason='requires torch')
    def test_gradgradient(self):
        gradgradcheck(
            FusedBiasLeakyReLU(2).cuda(),
            self.input_tensor,
            eps=1e-4,
            atol=1e-3)
