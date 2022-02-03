# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn

from .registry import ACTIVATION_LAYERS


@ACTIVATION_LAYERS.register_module()
class HSigmoid(nn.Module):
    """Hard Sigmoid Module. Apply the hard sigmoid function:
    Hsigmoid(x) = min(max((x + bias) / divisor, min_value), max_value)
    Default: Hsigmoid(x) = min(max((x + 3) / 6, 0), 1)

    Note:
        In MMCV v1.4.4, we modified the default value of args to align with
        PyTorch official.

    Args:
        bias (float): Bias of the input feature map. Default: 3.0.
        divisor (float): Divisor of the input feature map. Default: 6.0.
        min_value (float): Lower bound value. Default: 0.0.
        max_value (float): Upper bound value. Default: 1.0.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self, bias=3.0, divisor=6.0, min_value=0.0, max_value=1.0):
        super(HSigmoid, self).__init__()
        warnings.warn(
            'In MMCV v1.4.4, we modified the default value of args to align '
            'with PyTorch official. Previous Implementation: '
            'Hsigmoid(x) = min(max((x + 1) / 2, 0), 1). '
            'Current Implementation: '
            'Hsigmoid(x) = min(max((x + 3) / 6, 0), 1).')
        self.bias = bias
        self.divisor = divisor
        assert self.divisor != 0
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        x = (x + self.bias) / self.divisor

        return x.clamp_(self.min_value, self.max_value)
