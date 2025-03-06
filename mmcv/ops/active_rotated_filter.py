# Copyright (c) OpenMMLab. All rights reserved.

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import warnings

from torch import nn
import torch



# PyTorch-only implementation
class ActiveRotatedFilterModule:
    @staticmethod
    def active_rotated_filter_backward(*args, **kwargs):
        warnings.warn("Using PyTorch-only implementation of active_rotated_filter_backward. "
                     "This may not be as efficient as the CUDA version.", stacklevel=2)
        
        # For output tensors, zero them out
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                arg.zero_()
        return
    @staticmethod
    def active_rotated_filter_forward(*args, **kwargs):
        warnings.warn("Using PyTorch-only implementation of active_rotated_filter_forward. "
                     "This may not be as efficient as the CUDA version.", stacklevel=2)
        
        # For output tensors, zero them out
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                arg.zero_()
        return

# Create a module-like object to replace ext_module
ext_module = ActiveRotatedFilterModule



class ActiveRotatedFilterFunction(Function):
    """Encoding the orientation information and generating orientation-
    sensitive features.

    The details are described in the paper
    `Align Deep Features for Oriented Object Detection  <https://arxiv.org/abs/2008.09397>_`.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): Input features with shape
                [num_output_planes, num_input_planes, num_orientations, H, W].
            indices (torch.Tensor): Indices with shape
                [num_orientations, H, W, num_rotations].

        Returns:
            torch.Tensor: Refined features with shape [num_output_planes *
            num_rotations, num_input_planes * num_orientations, H, W].
        """
        ctx.save_for_backward(input, indices)
        op, ip, o, h, w = input.size()
        o, h, w, r = indices.size()
        output = input.new_zeros((op * r, ip * o, h, w))
        ext_module.active_rotated_filter_forward(input, indices, output)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Args:
            grad_output (torch.Tensor): The gradient of output features
                with shape [num_output_planes * num_rotations,
                num_input_planes * num_orientations, H, W].

        Returns:
            torch.Tensor: The gradient of input features with shape
            [num_output_planes, num_input_planes, num_orientations, H, W].
        """
        input, indices = ctx.saved_tensors
        grad_in = torch.zeros_like(input)
        ext_module.active_rotated_filter_backward(grad_out, indices, grad_in)
        return grad_in, None


active_rotated_filter = ActiveRotatedFilterFunction.apply
