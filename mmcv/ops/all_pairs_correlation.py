# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext',
    ['all_pairs_correlation_forward', 'all_pairs_correlation_backward'])
# from ..builder import OPERATORS


class AllPairsCorrelationFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2):
        ctx.save_for_backward(input1, input2)
        iH1, iW1 = input1.shape[-2:]
        iH2, iW2 = input2.shape[-2:]
        batch_size = input1.shape[0]
        output_size = (batch_size, iH1, iW1, iH2, iW2)

        output = input1.new_empty(output_size)

        ext_module.all_pairs_correlation_forward(input1, input2, output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        grad_input1 = torch.zeros_like(input1)
        grad_input2 = torch.zeros_like(input2)

        ext_module.all_pairs_correlation_backward(grad_output, input1, input2,
                                                  grad_input1, grad_input2)

        return grad_input1, grad_input2


# @OPERATORS.register_module()
class AllPairsCorrelation(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        return AllPairsCorrelationFunction.apply(input1, input2)

    def __repr__(self):
        s = self.__class__.__name__
        return s
