import torch
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', [
    'furthest_point_sampling_forward', 'stack_furthest_point_sampling_forward',
    'furthest_point_sampling_with_dist_forward'
])


class FurthestPointSampling(Function):
    """Uses iterative furthest point sampling to select a set of features whose
    corresponding points have the furthest distance."""

    @staticmethod
    def forward(ctx,
                points_xyz: torch.Tensor,
                num_points,
                points_batch_cnt=None) -> torch.Tensor:
        """
        Args:
            points_xyz (torch.Tensor): (B, N, 3) where N > num_points
                or stacked input (N1 + N2..., 3) .
            num_points (int): Number of points in the sampled set.
            points_batch_cnt (torch.Tensor): Stacked input points nums in
                each batch, just like (N1, N2, ...). Defaults to None.

        Returns:
            torch.Tensor: (B, num_points) indices of the sampled points.
        """
        assert points_xyz.is_contiguous()
        if points_batch_cnt is not None:
            assert points_xyz.shape[1] == 3
            B = len(points_batch_cnt)
            if not isinstance(num_points, torch.Tensor):
                if not isinstance(num_points, list):
                    num_points = [num_points for i in range(B)]
            num_points = torch.tensor(
                num_points, device=points_xyz.device).int().detach()
            N, _ = points_xyz.size()
            temp = torch.cuda.FloatTensor(N).fill_(1e10)
            output = torch.cuda.IntTensor(num_points.sum().item())
            ext_module.stack_furthest_point_sampling_forward(
                points_xyz, temp, points_batch_cnt, output, num_points)
        else:
            B, N = points_xyz.size()[:2]
            output = torch.cuda.IntTensor(B, num_points)
            temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

            ext_module.furthest_point_sampling_forward(
                points_xyz,
                temp,
                output,
                b=B,
                n=N,
                m=num_points,
            )
        if torch.__version__ != 'parrots':
            ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


class FurthestPointSamplingWithDist(Function):
    """Uses iterative furthest point sampling to select a set of features whose
    corresponding points have the furthest distance."""

    @staticmethod
    def forward(ctx, points_dist: torch.Tensor,
                num_points: int) -> torch.Tensor:
        """
        Args:
            points_dist (torch.Tensor): (B, N, N) Distance between each point
                pair.
            num_points (int): Number of points in the sampled set.

        Returns:
            torch.Tensor: (B, num_points) indices of the sampled points.
        """
        assert points_dist.is_contiguous()

        B, N, _ = points_dist.size()
        output = points_dist.new_zeros([B, num_points], dtype=torch.int32)
        temp = points_dist.new_zeros([B, N]).fill_(1e10)

        ext_module.furthest_point_sampling_with_dist_forward(
            points_dist, temp, output, b=B, n=N, m=num_points)
        if torch.__version__ != 'parrots':
            ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply
furthest_point_sample_with_dist = FurthestPointSamplingWithDist.apply
