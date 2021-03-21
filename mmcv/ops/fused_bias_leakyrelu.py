# modify from https://github.com/rosinality/stylegan2-pytorch/blob/master/op/fused_act.py # noqa:E501

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['fused_bias_leakyrelu'])


class FusedBiasLeakyReLUFunctionBackward(Function):
    """Calculate second order deviation.

    This function is to compute the second order deviation for the fused leaky
    relu operation.
    """

    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        grad_input = ext_module.fused_bias_leakyrelu(grad_output, empty, out,
                                                     3, 1, negative_slope,
                                                     scale)

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        grad_bias = grad_input.sum(dim).detach()

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors

        # The second order deviation, in fact, contains two parts, while the
        # the first part is zero. Thus, we direct consider the second part
        # which is similar with the first order deviation in implementation.
        gradgrad_out = ext_module.fused_bias_leakyrelu(gradgrad_input,
                                                       gradgrad_bias, out, 3,
                                                       1, ctx.negative_slope,
                                                       ctx.scale)

        return gradgrad_out, None, None, None


class FusedBiasLeakyReLUFunction(Function):

    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        out = ext_module.fused_bias_leakyrelu(input, bias, empty, 3, 0,
                                              negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors

        grad_input, grad_bias = FusedBiasLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.negative_slope, ctx.scale)

        return grad_input, grad_bias, None, None


class FusedBiasLeakyReLU(nn.Module):
    """Fused bias leaky ReLU.

    This function is introduced in the StyleGAN2:
    http://arxiv.org/abs/1912.04958

    The bias term comes from the convolution operation. In addition, to keep
    the variance of the feature map or gradients unchanged, they also adopt a
    scale similarly with Kaiming initalization. However, since the
    :math:`1 + \alpha^2` : is too small, we can just ignore it. Therefore, the
    final sacle is just :math:`\sqrt{2}`:. Of course, you may change it with # noqa: W605, E501
    your own scale.

    TODO: Implement the CPU version.

    Args:
        channel (int): The channnel number of the feature map.
        negative_slope (float, optional): Same as nn.LeakyRelu.
            Defaults to 0.2.
        scale (float, optional): A scalar to adjust the variance of the feature
            map. Defaults to 2**0.5.
    """

    def __init__(self, num_channels, negative_slope=0.2, scale=2**0.5):
        super(FusedBiasLeakyReLU, self).__init__()

        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_bias_leakyrelu(input, self.bias, self.negative_slope,
                                    self.scale)


def fused_bias_leakyrelu(input, bias, negative_slope=0.2, scale=2**0.5):
    """Fused bias leaky ReLU function.

    This function is introduced in the StyleGAN2:
    http://arxiv.org/abs/1912.04958

    The bias term comes from the convolution operation. In addition, to keep
    the variance of the feature map or gradients unchanged, they also adopt a
    scale similarly with Kaiming initalization. However, since the
    :math:`1 + \alpha^2` : is too small, we can just ignore it. Therefore, the
    final sacle is just :math:`\sqrt{2}`:. Of course, you may change it with # noqa: W605, E501
    your own scale.

    Args:
        input (torch.Tensor): Input feature map.
        bias (nn.Parameter): The bias from convolution operation.
        negative_slope (float, optional): Same as nn.LeakyRelu.
            Defaults to 0.2.
        scale (float, optional): A scalar to adjust the variance of the feature
            map. Defaults to 2**0.5.

    Returns:
        torch.Tensor: Feature map after non-linear activation.
    """

    if not input.is_cuda:
        return bias_leakyrelu_ref(input, bias, negative_slope, scale)

    return FusedBiasLeakyReLUFunction.apply(input, bias, negative_slope, scale)


def bias_leakyrelu_ref(x, bias, negative_slope=0.2, scale=2**0.5):

    if bias is not None:
        assert bias.ndim == 1
        assert bias.shape[0] == x.shape[1]
        x = x + bias.reshape([-1 if i == 1 else 1 for i in range(x.ndim)])

    x = F.leaky_relu(x, negative_slope)
    if scale != 1:
        x = x * scale

    return x
