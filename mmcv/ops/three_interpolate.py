from typing import Any
import warnings

import torch
from torch.autograd import Function

# PyTorch-only implementation
class ExtModule:
    @staticmethod
    def three_interpolate_forward(features, indices, weight, output, b, c, m, n):
        warnings.warn("Using PyTorch-only implementation of three_interpolate_forward. "
                     "This may not be as efficient as the CUDA version.", stacklevel=2)
                     
        # Implementation using PyTorch operations
        for batch_idx in range(b):
            for c_idx in range(c):
                for n_idx in range(n):
                    val = 0
                    for k in range(3):  # 3 nearest neighbors
                        val += weight[batch_idx, n_idx, k] * features[batch_idx, c_idx, indices[batch_idx, n_idx, k]]
                    output[batch_idx, c_idx, n_idx] = val
                    
        return output
    
    @staticmethod
    def three_interpolate_backward(grad_out, idx, weight, grad_features, b, c, n, m):
        warnings.warn("Using PyTorch-only implementation of three_interpolate_backward. "
                     "This may not produce correct gradients.", stacklevel=2)
                     
        # Implementation using PyTorch operations
        for batch_idx in range(b):
            for c_idx in range(c):
                for n_idx in range(n):
                    for k in range(3):  # 3 nearest neighbors
                        grad_features[batch_idx, c_idx, idx[batch_idx, n_idx, k]] += \
                            grad_out[batch_idx, c_idx, n_idx] * weight[batch_idx, n_idx, k]
                            
        return

# Create a module-like object to replace ext_module
ext_module = ExtModule


class ThreeInterpolate(Function):
    """Performs weighted linear interpolation on 3 features.

    Please refer to `Paper of PointNet++ <https://arxiv.org/abs/1706.02413>`_
    for more details.
    """

    @staticmethod
    def forward(ctx: Any, features: torch.Tensor, indices: torch.Tensor,
                weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): (B, C, M) Features descriptors to be
                interpolated.
            indices (torch.Tensor): (B, n, 3) indices of three nearest
                neighbor features for the target features.
            weight (torch.Tensor): (B, n, 3) weights of three nearest
                neighbor features for the target features.

        Returns:
            torch.Tensor: (B, C, N) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert indices.is_contiguous()
        assert weight.is_contiguous()

        B, c, m = features.size()
        n = indices.size(1)
        ctx.three_interpolate_for_backward = (indices, weight, m)
        output = features.new_empty(B, c, n)

        ext_module.three_interpolate_forward(
            features, indices, weight, output, b=B, c=c, m=m, n=n)
        return output

    @staticmethod
    def backward(
        ctx, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            grad_out (torch.Tensor): (B, C, N) tensor with gradients of outputs

        Returns:
            torch.Tensor: (B, C, M) tensor with gradients of features
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        grad_features = grad_out.new_zeros(B, c, m)
        grad_out_data = grad_out.data.contiguous()

        ext_module.three_interpolate_backward(
            grad_out_data, idx, weight, grad_features.data, b=B, c=c, n=n, m=m)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply
