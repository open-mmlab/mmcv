# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.ops import filtered_lrelu


class TestFilteredLrelu:

    @classmethod
    def setup_class(cls):
        if not torch.cuda.is_available():
            return
        cls.input_tensor = torch.randn((1, 3, 16, 16), requires_grad=True)
        cls.bias = torch.randn(3, requires_grad=True)
        cls.fu = torch.randn((2, 2))
        cls.fd = torch.randn((2, 2))

    def test_filtered_lrelu_cpu(self):
        out = filtered_lrelu(self.input_tensor, b=self.bias)
        assert out.shape == (1, 3, 16, 16)

        out = filtered_lrelu(
            self.input_tensor,
            b=self.bias,
            fu=self.fu,
            fd=self.fd,
            up=2,
            down=2,
            padding=1,
            clamp=0.5)
        assert out.shape == (1, 3, 16, 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_filtered_lrelu_cuda(self):
        out = filtered_lrelu(self.input_tensor.cuda(), b=self.bias.cuda())
        assert out.shape == (1, 3, 16, 16)
