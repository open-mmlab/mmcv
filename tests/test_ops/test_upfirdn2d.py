import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck


class TestUpFirDn2d(object):
    """Unit test for UpFirDn2d.

    Here, we just test the basic case of upsample version. More gerneal tests
    will be included in other unit test for UpFirDnUpsample and
    UpFirDnDownSample modules.
    """

    @classmethod
    def setup_class(cls):
        kernel_1d = torch.tensor([1., 3., 3., 1.])
        cls.kernel = kernel_1d[:, None] * kernel_1d[None, :]
        cls.kernel = cls.kernel / cls.kernel.sum()
        cls.factor = 2
        pad = cls.kernel.shape[0] - cls.factor
        cls.pad = ((pad + 1) // 2 + cls.factor - 1, pad // 2)

        cls.input_tensor = torch.randn((2, 3, 4, 4), requires_grad=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_upfirdn2d(self):
        from mmcv.ops import upfirdn2d

        gradcheck(
            upfirdn2d,
            (self.input_tensor.cuda(), self.kernel.type_as(
                self.input_tensor).cuda(), self.factor, 1, self.pad),
            eps=1e-4,
            atol=1e-3)

        gradgradcheck(
            upfirdn2d,
            (self.input_tensor.cuda(), self.kernel.type_as(
                self.input_tensor).cuda(), self.factor, 1, self.pad),
            eps=1e-4,
            atol=1e-3)
