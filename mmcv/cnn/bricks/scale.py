# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn


class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class LayerScale(nn.Module):
    """LayerScale layer.

    Args:
        dim (int): Dimension of input features.
        inplace (bool): Whether performs operation in-place.
            Default: `False`.
        data_format (str): The input data format, could be 'channels_last'
            or 'channels_first', representing (B, C, H, W) and
            (B, N, C) format data respectively. Default: 'channels_last'.
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self,
                 dim: int,
                 inplace: bool = False,
                 data_format: str = 'channels_last',
                 scale: float = 1e-5):
        super().__init__()
        assert data_format in ('channels_last', 'channels_first'), \
            "'data_format' could only be channels_last or channels_first."
        self.inplace = inplace
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(dim) * scale)

    def forward(self, x) -> torch.Tensor:
        if self.data_format == 'channels_first':
            shape = tuple((1, -1, *(1 for _ in range(x.dim() - 2))))
        else:
            shape = tuple((*(1 for _ in range(x.dim() - 1)), -1))
        if self.inplace:
            return x.mul_(self.weight.view(*shape))
        else:
            return x * self.weight.view(*shape)
