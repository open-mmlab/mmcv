from typing import Any, Tuple

import torch
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['three_interpolate_forward', 'three_interpolate_backward',
             'stack_three_interpolate_forward', 'stack_three_interpolate_backward'])


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
        output = torch.cuda.FloatTensor(B, c, n)

        ext_module.three_interpolate_forward(
            features, indices, weight, output, b=B, c=c, m=m, n=n)
        return output

    @staticmethod
    def backward(
        ctx, grad_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            grad_out (torch.Tensor): (B, C, N) tensor with gradients of outputs

        Returns:
            torch.Tensor: (B, C, M) tensor with gradients of features
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        grad_features = torch.cuda.FloatTensor(B, c, m).zero_()
        grad_out_data = grad_out.data.contiguous()

        ext_module.three_interpolate_backward(
            grad_out_data, idx, weight, grad_features.data, b=B, c=c, n=n, m=m)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply
class StackThreeInterpolate(Function):
    """Performs weighted linear interpolation on 3 features.

    Please refer to `Paper of PointNet++ <https://arxiv.org/abs/1706.02413>`_
    for more details.
    """

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor):
        """
        Args:
            ctx:
            features: (M1 + M2 ..., C)
            idx: [N1 + N2 ..., 3]
            weight: [N1 + N2 ..., 3]

        Returns:
            out_tensor: (N1 + N2 ..., C)
        """
        assert idx.shape[0] == weight.shape[0] and idx.shape[1] == weight.shape[1] == 3

        ctx.three_interpolate_for_backward = (idx, weight, features.shape[0])
        output = features.new_zeros((idx.shape[0], features.shape[1]))
        ext_module.stack_three_interpolate_forward(features.contiguous(), idx.contiguous(), weight.contiguous(), output)
        return output
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: (N1 + N2 ..., C)

        Returns:
            grad_features: (M1 + M2 ..., C)
        """
        idx, weight, M = ctx.three_interpolate_for_backward
        grad_features = grad_out.new_zeros((M, grad_out.shape[1]))
        ext_module.stack_three_interpolate_backward(
            grad_out.contiguous(), idx.contiguous(), weight.contiguous(), grad_features
        )
        return grad_features, None, None


stack_three_interpolate = StackThreeInterpolate.apply
