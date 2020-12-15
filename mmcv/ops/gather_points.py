import torch
from torch.autograd import Function

from ..utils import ext_loader

if torch.__version__ == 'parrots':
    load_ext = '_ext_pt'
else:
    load_ext = '_ext'

ext_module = ext_loader.load_ext(load_ext,
                                 ['gather_points', 'gather_points_backward'])


class GatherPoints(Function):
    """Gather Points.

    Gather points with given index.
    """

    @staticmethod
    def forward(ctx, features: torch.Tensor,
                indicies: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            features (Tensor): (B, C, N) features to gather.
            indicies (Tensor): (B, M) where M is the number of points.

        Returns:
            Tensor: (B, C, M) where M is the number of points.
        """
        assert features.is_contiguous()
        assert indicies.is_contiguous()

        B, npoint = indicies.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint)

        if torch.__version__ == 'parrots':
            indata_list = [features, indicies, output]
            indata_dict = {'b': B, 'c': C, 'n': N, 'npoints': npoint}
            ext_module.gather_points(*indata_list, **indata_dict)
        else:
            ext_module.gather_points(B, C, N, npoint, features, indicies,
                                     output)

        ctx.for_backwards = (indicies, C, N)
        if torch.__version__ != 'parrots':
            ctx.mark_non_differentiable(indicies)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = torch.cuda.FloatTensor(B, C, N).zero_()
        grad_out_data = grad_out.data.contiguous()
        if torch.__version__ == 'parrots':
            indata_list = [grad_out_data, idx, grad_features.data]
            indata_dict = {'b': B, 'c': C, 'n': N, 'npoints': npoint}
            ext_module.gather_points_backward(*indata_list, **indata_dict)
        else:
            ext_module.gather_points_backward(B, C, N, npoint, grad_out_data,
                                              idx, grad_features.data)
        return grad_features, None


gather_points = GatherPoints.apply
