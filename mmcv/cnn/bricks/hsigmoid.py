import torch.nn as nn

from .registry import ACTIVATION_LAYERS


@ACTIVATION_LAYERS.register_module()
class HSigmoid(nn.Module):
    """Hard Sigmoid Module. Apply the hard sigmoid function:
    Hsigmoid(x) = min(max((x + bias) / divisor, min_value), max_value)
    Hsigmoid(x) = min(max((x + 1) / 2, 0), 1) by default

    Args:
        bias (int | float): Bias of the input feature map. Default: 1.
        divisor (int | float): Divisor of the input feature map. Default: 2.
        min_value (int | float): Lower bound value. Default: 0.
        max_value (int | float): Upper bound value. Default: 1.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self, bias=1, divisor=2, min_value=0, max_value=1):
        super(HSigmoid, self).__init__()
        self.bias = bias
        self.divisor = divisor
        assert self.divisor != 0
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        x = (x + self.bias) / self.divisor

        return x.clamp_(self.min_value, self.max_value)
