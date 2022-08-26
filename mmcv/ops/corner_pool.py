# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor, nn

_mode_dict = {'top': 0, 'bottom': 1, 'left': 2, 'right': 3}


def _corner_pool(x: Tensor, dim: int, flip: bool) -> Tensor:
    size = x.size(dim)
    output = x.clone()

    ind = 1
    while ind < size:
        if flip:
            cur_start = 0
            cur_len = size - ind
            next_start = ind
            next_len = size - ind
        else:
            cur_start = ind
            cur_len = size - ind
            next_start = 0
            next_len = size - ind

        # max_temp should be cloned for backward computation
        max_temp = output.narrow(dim, cur_start, cur_len).clone()
        cur_temp = output.narrow(dim, cur_start, cur_len)
        next_temp = output.narrow(dim, next_start, next_len)

        cur_temp[...] = torch.where(max_temp > next_temp, max_temp, next_temp)

        ind = ind << 1

    return output


class CornerPool(nn.Module):
    """Corner Pooling.

    Corner Pooling is a new type of pooling layer that helps a
    convolutional network better localize corners of bounding boxes.

    Please refer to `CornerNet: Detecting Objects as Paired Keypoints
    <https://arxiv.org/abs/1808.01244>`_ for more details.

    Code is modified from https://github.com/princeton-vl/CornerNet-Lite.

    Args:
        mode (str): Pooling orientation for the pooling layer

            - 'bottom': Bottom Pooling
            - 'left': Left Pooling
            - 'right': Right Pooling
            - 'top': Top Pooling

    Returns:
        Feature map after pooling.
    """

    cummax_dim_flip = {
        'bottom': (2, False),
        'left': (3, True),
        'right': (3, False),
        'top': (2, True),
    }

    def __init__(self, mode: str):
        super().__init__()
        assert mode in self.cummax_dim_flip
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        if torch.__version__ != 'parrots' and torch.__version__ >= '1.5.0':
            dim, flip = self.cummax_dim_flip[self.mode]
            if flip:
                x = x.flip(dim)
            pool_tensor, _ = torch.cummax(x, dim=dim)
            if flip:
                pool_tensor = pool_tensor.flip(dim)
            return pool_tensor
        else:
            dim, flip = self.cummax_dim_flip[self.mode]
            return _corner_pool(x, dim, flip)
