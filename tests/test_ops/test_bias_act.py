# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.ops import bias_act


class TestBiasAct:

    @classmethod
    def setup_class(cls):
        cls.input_tensor = torch.randn((1, 3), requires_grad=True)
        cls.bias = torch.randn(3, requires_grad=True)

    def test_bias_act_cpu(self):
        out = bias_act(self.input_tensor, self.bias)
        assert out.shape == (1, 3)

        # test with different dim
        input_tensor = torch.randn((1, 1, 3), requires_grad=True)
        bias = torch.randn(3, requires_grad=True)
        out = bias_act(input_tensor, bias, dim=2)
        assert out.shape == (1, 1, 3)

        # test with different act
        out = bias_act(self.input_tensor, self.bias, act='relu')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor, self.bias, act='lrelu')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor, self.bias, act='tanh')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor, self.bias, act='sigmoid')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor, self.bias, act='elu')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor, self.bias, act='selu')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor, self.bias, act='softplus')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor, self.bias, act='swish')
        assert out.shape == (1, 3)

        # test with different alpha
        out = bias_act(self.input_tensor, self.bias, act='lrelu', alpha=0.1)
        assert out.shape == (1, 3)

        # test with different gain
        out1 = bias_act(self.input_tensor, self.bias, act='lrelu', gain=0.2)
        out2 = bias_act(self.input_tensor, self.bias, act='lrelu', gain=0.1)
        assert torch.allclose(out1, out2 * 2)

        # test with different clamp
        out1 = bias_act(self.input_tensor, self.bias, act='lrelu', clamp=0.5)
        out2 = bias_act(self.input_tensor, self.bias, act='lrelu', clamp=0.2)
        assert out1.max() <= 0.5
        assert out2.max() <= 0.5

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_bias_act_cuda(self):
        out = bias_act(self.input_tensor.cuda(), self.bias.cuda())
        assert out.shape == (1, 3)

        # test with different dim
        input_tensor = torch.randn((1, 1, 3), requires_grad=True).cuda()
        bias = torch.randn(3, requires_grad=True).cuda()
        out = bias_act(input_tensor, bias, dim=2)
        assert out.shape == (1, 1, 3)

        # test with different act
        out = bias_act(self.input_tensor.cuda(), self.bias.cuda(), act='relu')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor.cuda(), self.bias.cuda(), act='lrelu')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor.cuda(), self.bias.cuda(), act='tanh')
        assert out.shape == (1, 3)
        out = bias_act(
            self.input_tensor.cuda(), self.bias.cuda(), act='sigmoid')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor.cuda(), self.bias.cuda(), act='elu')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor.cuda(), self.bias.cuda(), act='selu')
        assert out.shape == (1, 3)
        out = bias_act(
            self.input_tensor.cuda(), self.bias.cuda(), act='softplus')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor.cuda(), self.bias.cuda(), act='swish')
        assert out.shape == (1, 3)

        # test with different alpha
        out = bias_act(
            self.input_tensor.cuda(), self.bias.cuda(), act='lrelu', alpha=0.1)
        assert out.shape == (1, 3)

        # test with different gain
        out1 = bias_act(
            self.input_tensor.cuda(), self.bias.cuda(), act='lrelu', gain=0.2)
        out2 = bias_act(
            self.input_tensor.cuda(), self.bias.cuda(), act='lrelu', gain=0.1)
        assert torch.allclose(out1, out2 * 2)

        # test with different clamp
        out1 = bias_act(
            self.input_tensor.cuda(), self.bias.cuda(), act='lrelu', clamp=0.5)
        out2 = bias_act(
            self.input_tensor.cuda(), self.bias.cuda(), act='lrelu', clamp=0.2)
        assert out1.max() <= 0.5
        assert out2.max() <= 0.5
