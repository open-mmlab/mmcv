# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['correlation_forward', 'correlation_backward'])
# from ..builder import OPERATORS


class CorrelationFunction(Function):

    @staticmethod
    def forward(ctx,
                input1,
                input2,
                kernel_size=1,
                max_displacement=1,
                stride=1,
                padding=1,
                dilation=1,
                dilation_patch=1):

        ctx.save_for_backward(input1, input2)

        kH, kW = ctx.kernel_size = _pair(kernel_size)
        patch_size = max_displacement * 2 + 1
        ctx.patch_size = patch_size
        dH, dW = ctx.stride = _pair(stride)
        padH, padW = ctx.padding = _pair(padding)
        dilationH, dilationW = ctx.dilation = _pair(dilation)
        dilation_patchH, dilation_patchW = ctx.dilation_patch = _pair(
            dilation_patch)

        output_size = CorrelationFunction._output_size(ctx, input1)

        output = input1.new_empty(output_size)

        ext_module.correlation_forward(input1, input2, output, kH, kW,
                                       patch_size, patch_size, padH, padW,
                                       dilationH, dilationW, dilation_patchH,
                                       dilation_patchW, dH, dW)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors

        kH, kW = ctx.kernel_size
        patch_size = ctx.patch_size
        padH, padW = ctx.padding
        dilationH, dilationW = ctx.dilation
        dilation_patchH, dilation_patchW = ctx.dilation_patch
        dH, dW = ctx.stride
        grad_input1 = torch.zeros_like(input1)
        grad_input2 = torch.zeros_like(input2)

        ext_module.correlation_backward(grad_output, input1, input2,
                                        grad_input1, grad_input2, kH, kW,
                                        patch_size, patch_size, padH, padW,
                                        dilationH, dilationW, dilation_patchH,
                                        dilation_patchW, dH, dW)
        return grad_input1, grad_input2, None, None, None, None, None, None

    @staticmethod
    def _output_size(ctx, input1):
        iH, iW = input1.size(2), input1.size(3)
        batch_size = input1.size(0)
        kH, kW = ctx.kernel_size
        patch_size = ctx.patch_size
        dH, dW = ctx.stride
        padH, padW = ctx.padding
        dilationH, dilationW = ctx.dilation
        dilatedKH = (kH - 1) * dilationH + 1
        dilatedKW = (kW - 1) * dilationW + 1

        oH = int((iH + 2 * padH - dilatedKH) / dH + 1)
        oW = int((iW + 2 * padW - dilatedKW) / dW + 1)

        output_size = (batch_size, patch_size, patch_size, oH, oW)
        return output_size


# @OPERATORS.register_module()
class Correlation(nn.Module):
    """[summary]

    Args:
        kernel_size (int, optional): [description]. Defaults to 1.
        max_displacement (int, optional): the max displacement i.e. the radius
            for computing cost volume . Defaults to 1.
        stride (int, optional): [description]. Defaults to 1.
        padding (int, optional): [description]. Defaults to 0.
        dilation (int, optional): [description]. Defaults to 1.
        dilation_patch (int, optional): [description]. Defaults to 1.
    """

    def __init__(self,
                 kernel_size=1,
                 max_displacement=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 dilation_patch=1):

        super().__init__()
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dilation_patch = dilation_patch

    def forward(self, input1, input2):
        return CorrelationFunction.apply(input1, input2, self.kernel_size,
                                         self.max_displacement, self.stride,
                                         self.padding, self.dilation,
                                         self.dilation_patch)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(kernel_size={self.kernel_size}, '
        s += f'max_displacement={self.max_displacement}, '
        s += f'stride={self.stride}, '
        s += f'padding={self.padding}, '
        s += f'dilation={self.dilation}, '
        s += f'dilation_patch={self.dilation_patch})'
        return s
