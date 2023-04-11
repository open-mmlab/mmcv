# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmengine.model.weight_init import trunc_normal_
from torch import nn
from torch.nn.functional import pad

from .nattenav_functional import nattenav
from .nattenqkrpb_functional import nattenqkrpb


class NeighborhoodAttention(nn.Module):
    """Neighborhood Attention Module.

    Args:
        dim (int): Dimension of the input.
        kernel_size (int): Kernel size of the neighborhood attention.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): Whether to add bias to the query,
            key, and value. Defaults to True.
        qk_scale (Optional[float], optional): Scale factor for the
            query and key. Defaults to None.
        attn_drop (float, optional): Dropout rate for the attention weights.
            Defaults to 0.
        proj_drop (float, optional): Dropout rate for the output.
            Defaults to 0.
    """

    def __init__(self,
                 dim: int,
                 kernel_size: int,
                 num_heads: int,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f'CUDA kernel only supports kernel sizes' \
            f' 3, 5, 7, 9, 11, and 13; got {kernel_size}.'
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(
            torch.zeros(num_heads, (2 * kernel_size - 1),
                        (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads,
                                  self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = nattenqkrpb(q, k, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = nattenav(attn, v)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))
