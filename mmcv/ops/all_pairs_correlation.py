# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext',
    ['all_pairs_correlation_forward', 'all_pairs_correlation_backward'])


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


class AllPairsCorrelation(nn.Module):
    """All-pairs correlation operator.

    All-pairs correlation is used to compute visual similarity in `RAFT
        <https://link.springer.com/chapter/10.1007/978-3-030-58536-5_24>`_.
    The correlation output with the shape (N, H, W, H, W) is formed by taking
    the dot product between all pairs of input feature vectors , and H, W is
    equal the input tensors'.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        return AllPairsCorrelationFunction.apply(input1, input2)

    def __repr__(self) -> str:
        s = self.__class__.__name__
        return s
