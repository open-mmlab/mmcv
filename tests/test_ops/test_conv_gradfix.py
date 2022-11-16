from functools import partial

import pytest
import torch
import torch.nn as nn
from torch.autograd import gradgradcheck

from mmcv.ops import conv2d, conv_transpose2d


class TestCond2d:

    @classmethod
    def setup_class(cls):
        cls.input = torch.randn((1, 3, 32, 32))
        cls.weight = nn.Parameter(torch.randn(1, 3, 3, 3))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_conv2d_cuda(self):
        x = self.input.cuda()
        weight = self.weight.cuda()
        res = conv2d(x, weight, None, 1, 1)
        assert res.shape == (1, 1, 32, 32)
        gradgradcheck(conv2d, (x, weight, None, 1, 1), eps=1e-2, atol=1e-2)


class TestCond2dTansposed:

    @classmethod
    def setup_class(cls):
        cls.input = torch.randn((1, 3, 32, 32))
        cls.weight = nn.Parameter(torch.randn(3, 1, 3, 3))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_conv2d_transposed_cuda(self):
        x = self.input.cuda()
        weight = self.weight.cuda()
        res = conv_transpose2d(x, weight, None, 1, 1)
        assert res.shape == (1, 1, 32, 32)
        gradgradcheck(
            conv_transpose2d, (x, weight, None, 1, 1), eps=1e-2, atol=1e-2)
