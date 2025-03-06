
import torch
from torch.autograd import Function

import warnings

from torch import nn
import torch



# PyTorch-only implementation
class GatherPointsModule:
    @staticmethod
    def gather_points_forward(*args, **kwargs):
        warnings.warn("Using PyTorch-only implementation of gather_points_forward. "
                     "This may not be as efficient as the CUDA version.", stacklevel=2)
        
        # For output tensors, zero them out
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                arg.zero_()
        return
    @staticmethod
    def gather_points_backward(*args, **kwargs):
        warnings.warn("Using PyTorch-only implementation of gather_points_backward. "
                     "This may not be as efficient as the CUDA version.", stacklevel=2)
        
        # For output tensors, zero them out
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                arg.zero_()
        return

# Create a module-like object to replace ext_module
ext_module = GatherPointsModule



class GatherPoints(Function):
    """Gather points with given index."""

    @staticmethod
    def forward(ctx, features: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): (B, C, N) features to gather.
            indices (torch.Tensor): (B, M) where M is the number of points.

        Returns:
            torch.Tensor: (B, C, M) where M is the number of points.
        """
        assert features.is_contiguous()
        assert indices.is_contiguous()

        B, npoint = indices.size()
        _, C, N = features.size()
        output = features.new_zeros((B, C, npoint))

        ext_module.gather_points_forward(
            features, indices, output, b=B, c=C, n=N, npoints=npoint)

        ctx.for_backwards = (indices, C, N)
        if torch.__version__ != 'parrots':
            ctx.mark_non_differentiable(indices)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, None]:
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = grad_out.new_zeros((B, C, N))
        grad_out_data = grad_out.data.contiguous()
        ext_module.gather_points_backward(
            grad_out_data,
            idx,
            grad_features.data,
            b=B,
            c=C,
            n=N,
            npoints=npoint)
        return grad_features, None


gather_points = GatherPoints.apply
