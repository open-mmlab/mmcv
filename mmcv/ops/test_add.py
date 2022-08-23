import torch 
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ..utils import ext_loader
ext_module = ext_loader.load_ext('_ext', ['test_add'])


class TestAdd(Function):

    @staticmethod
    def forward(ctx, input1: torch.Tensor,
                input2: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(
            input1.size(0), dtype=input1.dtype, device=input1.device
        )
        ext_module.test_add(input1, input2, output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return grad_output, grad_output

test_add = TestAdd.apply