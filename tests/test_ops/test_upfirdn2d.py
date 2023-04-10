# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch


from torch.autograd import gradcheck, gradgradcheck


class TestUpFirDn2d:
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
            (self.input_tensor.cuda(),
                self.kernel.type_as(
                    self.input_tensor).cuda(), self.factor, 1, self.pad),
            eps=1e-4,
            atol=1e-3)

        gradgradcheck(
            upfirdn2d,
            (self.input_tensor.cuda(),
                self.kernel.type_as(
                    self.input_tensor).cuda(), self.factor, 1, self.pad),
            eps=1e-4,
            atol=1e-3)

        # test with different up
        kernel = torch.randn(3, 3)
        out = upfirdn2d(
            self.input_tensor.cuda(), filter=kernel.cuda(), up=2, padding=1)
        assert out.shape == (2, 3, 8, 8)

        # test with different down
        input_tensor = torch.randn(2, 3, 8, 8)
        out = upfirdn2d(
            input_tensor.cuda(), filter=self.kernel.cuda(), down=2, padding=1)
        assert out.shape == (2, 3, 4, 4)

        # test with different flip_filter
        out = upfirdn2d(
            self.input_tensor.cuda(),
            filter=self.kernel.cuda(),
            flip_filter=True)
        assert out.shape == (2, 3, 1, 1)

        # test with different gain
        out1 = upfirdn2d(
            self.input_tensor.cuda(), filter=self.kernel.cuda(), gain=0.2)
        out2 = upfirdn2d(
            self.input_tensor.cuda(), filter=self.kernel.cuda(), gain=0.1)
        assert torch.allclose(out1, out2 * 2)
