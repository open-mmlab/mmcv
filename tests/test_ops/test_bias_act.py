# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.ops import bias_act


class TestBiasAct:

    @classmethod
    def setup_class(cls):
        if not torch.cuda.is_available():
            return
        cls.input_tensor = torch.randn((1, 3), requires_grad=True).cuda()
        cls.bias = torch.randn(3, requires_grad=True).cuda()

    def test_bias_act_cpu(self):
        out = bias_act(self.input_tensor, self.bias)
        assert out.shape == (1, 3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_bias_act_cuda(self):
        out = bias_act(self.input_tensor.cuda(), self.bias.cuda())
        assert out.shape == (1, 3)
