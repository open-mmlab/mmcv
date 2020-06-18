import torch.nn as nn

from .registry import ACTIVATION_LAYERS


@ACTIVATION_LAYERS.register_module()
class HSigmoid(nn.Module):
    """Hard Sigmoid Module. Apply the hard sigmoid function:
    Hsigmoid(x) = min(max(0, (x + 1) / 2), 1)

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self):
        super(HSigmoid, self).__init__()

    @staticmethod
    def forward(x):
        x = (x + 1) / 2

        return x.clamp_(0, 1)
