from typing import Tuple

import torch
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['three_nn', 'three_interpolate', 'three_interpolate_backward'])


class ThreeInterpolate(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, indices: torch.Tensor,
                weight: torch.Tensor) -> torch.Tensor:
        """Performs weighted linear interpolation on 3 features.

        Args:
            features (Tensor): (B, C, M) Features descriptors to be
                interpolated from
            indices (Tensor): (B, n, 3) index three nearest neighbors
                of the target features in features
            weight (Tensor): (B, n, 3) weights of interpolation

        Returns:
            Tensor: (B, C, N) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert indices.is_contiguous()
        assert weight.is_contiguous()

        B, c, m = features.size()
        n = indices.size(1)
        ctx.three_interpolate_for_backward = (indices, weight, m)
        output = torch.cuda.FloatTensor(B, c, n)

        ext_module.three_interpolate(B, c, m, n, features, indices, weight,
                                     output)
        return output

    @staticmethod
    def backward(
        ctx, grad_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward of three interpolate.

        Args:
            grad_out (Tensor): (B, C, N) tensor with gradients of outputs

        Returns:
            Tensor: (B, C, M) tensor with gradients of features
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        grad_features = torch.cuda.FloatTensor(B, c, m).zero_()
        grad_out_data = grad_out.data.contiguous()

        ext_module.three_interpolate_backward(B, c, n, m, grad_out_data, idx,
                                              weight, grad_features.data)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, target: torch.Tensor,
                source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find the top-3 nearest neighbors of the target set from the source
        set.

        Args:
            target (Tensor): shape (B, N, 3), points set that needs to
                find the nearest neighbors.
            source (Tensor): shape (B, M, 3), points set that is used
                to find the nearest neighbors of points in target set.

        Returns:
            Tensor: shape (B, N, 3), L2 distance of each point in target
                set to their corresponding nearest neighbors.
        """
        assert target.is_contiguous()
        assert source.is_contiguous()

        B, N, _ = target.size()
        m = source.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)

        ext_module.three_nn(B, N, m, target, source, dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply
