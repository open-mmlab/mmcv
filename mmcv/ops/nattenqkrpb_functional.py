# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['nattenqkrpb_forward', 'nattenqkrpb_backward'])


class NATTENQKRPBFunction(Function):
    """QK+RPB autograd function Computes neighborhood attention weights given
    queries and keys, and adds relative positional biases.

    This calls the `QKRPB` kernel.
    """

    @staticmethod
    def forward(ctx, query: torch.Tensor, key: torch.Tensor,
                rpb: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            query (torch.Tensor): Queries.
            key (torch.Tensor): Keys.
            rpb (torch.Tensor): Relative positional biases.

        Returns:
            torch.Tensor: Attention weights.
        """
        query = query.contiguous()
        key = key.contiguous()
        attn = ext_module.nattenqkrpb_forward(query, key, rpb)
        ctx.save_for_backward(query, key)
        return attn

    @staticmethod
    def backward(
        ctx, grad_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Backward function.

        Args:
            grad_out (torch.Tensor): Gradient of the output.

        Returns:
            Tuple:
                - torch.Tensor: Gradient of the query.
                - torch.Tensor: Gradient of the key.
                - torch.Tensor: Gradient of the rpb.
                - None: Dummy variable.
        """
        outputs = ext_module.nattenqkrpb_backward(grad_out.contiguous(),
                                                  ctx.saved_tensors[0],
                                                  ctx.saved_tensors[1])
        d_query, d_key, d_rpb = outputs
        return d_query, d_key, d_rpb, None


nattenqkrpb = NATTENQKRPBFunction.apply
