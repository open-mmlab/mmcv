import torch.nn as nn

from .registry import ACTIVATION_LAYERS


@ACTIVATION_LAYERS.register_module()
class Swish(nn.Module):
    """ wish Module.

    This module applies the swish function:

    .. math::
        Swish(x) = x * Sigmoid(x)

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self, inplace):
        super(Swish, self).__init__()
        self.inplace = inplace
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(x)
