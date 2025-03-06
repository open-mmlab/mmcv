from typing import Any

import torch
from torch.autograd import Function

from mmcv.ops.pure_pytorch_three_nn.three_nn_forward import three_nn_forward_pytorch


class ThreeNN(Function):
    """Find the top-3 nearest neighbors of the target set from the source set.

    Please refer to `Paper of PointNet++ <https://arxiv.org/abs/1706.02413>`_
    for more details.
    """

    @staticmethod
    def forward(ctx: Any, target: torch.Tensor,
                source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            target (torch.Tensor): shape (B, N, 3), points set that needs to
                find the nearest neighbors.
            source (torch.Tensor): shape (B, M, 3), points set that is used
                to find the nearest neighbors of points in target set.

        Returns:
            torch.Tensor: shape (B, N, 3), L2 distance of each point in target
            set to their corresponding top three nearest neighbors.
        """
        target = target.contiguous()
        source = source.contiguous()

        B, N, _ = target.size()
        m = source.size(1)
        dist2 = target.new_empty(B, N, 3)
        idx = target.new_empty(B, N, 3, dtype=torch.int32)

        three_nn_forward_pytorch(target, source, dist2, idx, b=B, n=N, m=m)
        if torch.__version__ != 'parrots':
            ctx.mark_non_differentiable(idx)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply
