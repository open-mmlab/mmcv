# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext',
                                 ['nattenav_forward', 'nattenav_backward'])


class NATTENAVFunction(Function):
    """AV autograd function Computes neighborhood attention outputs given
    attention weights, and values.

    This calls the `AV` kernel.
    """

    @staticmethod
    def forward(ctx, attn: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            attn (torch.Tensor): Attention weights.
            value (torch.Tensor): Values.

        Returns:
            torch.Tensor: Attention outputs.
        """
        attn = attn.contiguous()
        value = value.contiguous()
        out = ext_module.nattenav_forward(attn, value)
        ctx.save_for_backward(attn, value)
        return out

    @staticmethod
    def backward(ctx,
                 grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward function.

        Args:
            grad_out (torch.Tensor): Gradient of the output.

        Returns:
            Tuple:
                - torch.Tensor: Gradient of the attention weights.
                - torch.Tensor: Gradient of the values.
        """
        outputs = ext_module.nattenav_backward(grad_out.contiguous(),
                                               ctx.saved_tensors[0],
                                               ctx.saved_tensors[1])
        d_attn, d_value = outputs
        return d_attn, d_value


nattenav = NATTENAVFunction.apply
