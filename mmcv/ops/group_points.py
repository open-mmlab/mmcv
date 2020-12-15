from typing import Tuple

import torch
from torch.autograd import Function

from ..utils import ext_loader

if torch.__version__ == 'parrots':
    ext_load = '_ext_pt'
else:
    ext_load = '_ext'

ext_module = ext_loader.load_ext(ext_load,
                                 ['group_points', 'group_points_backward'])


class GroupingOperation(Function):
    """Grouping Operation.

    Group feature with given index.
    """

    @staticmethod
    def forward(ctx, features: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            features (Tensor): (B, C, N) tensor of features to group.
            indices (Tensor): (B, npoint, nsample) the indicies of
                features to group with.

        Returns:
            Tensor: (B, C, npoint, nsample) Grouped features.
        """
        assert features.is_contiguous()
        assert indices.is_contiguous()

        B, nfeatures, nsample = indices.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample)

        if torch.__version__ == 'parrots':
            indata_list = [features, indices, output]
            indata_dict = {
                'b': B,
                'c': C,
                'n': N,
                'npoints': nfeatures,
                'nsample': nsample
            }
            ext_module.group_points(*indata_list, **indata_dict)
        else:
            ext_module.group_points(B, C, N, nfeatures, nsample, features,
                                    indices, output)

        ctx.for_backwards = (indices, N)
        return output

    @staticmethod
    def backward(ctx,
                 grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """backward.

        Args:
            grad_out (Tensor): (B, C, npoint, nsample) tensor of the gradients
                of the output from forward.

        Returns:
            Tensor: (B, C, N) gradient of the features.
        """
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = grad_out.size()
        grad_features = torch.cuda.FloatTensor(B, C, N).zero_()

        grad_out_data = grad_out.data.contiguous()
        if torch.__version__ == 'parrots':
            indata_list = [grad_out_data, idx, grad_features.data]
            indata_dict = {
                'b': B,
                'c': C,
                'n': N,
                'npoints': npoint,
                'nsample': nsample
            }
            ext_module.group_points_backward(*indata_list, **indata_dict)
        else:
            ext_module.group_points_backward(B, C, N, npoint, nsample,
                                             grad_out_data, idx,
                                             grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply
