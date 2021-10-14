import torch
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['gather_points_forward', 'gather_points_backward'])


class GatherPoints(Function):
    """Gather points with given index."""

    @staticmethod
    def forward(ctx, features: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (Tensor): (B, C, N) features to gather.
            indices (Tensor): (B, M) where M is the number of points.

        Returns:
            Tensor: (B, C, M) where M is the number of points.
        """
        assert features.is_contiguous()
        assert indices.is_contiguous()

        B, npoint = indices.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint)

        ext_module.gather_points_forward(B, C, N, npoint, features, indices,
                                         output)

        ctx.for_backwards = (indices, C, N)
        ctx.mark_non_differentiable(indices)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = torch.cuda.FloatTensor(B, C, N).zero_()
        grad_out_data = grad_out.data.contiguous()
        ext_module.gather_points_backward(B, C, N, npoint, grad_out_data, idx,
                                          grad_features.data)
        return grad_features, None


gather_points = GatherPoints.apply
