# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

for module in [
        nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.RReLU, nn.ReLU6, nn.ELU,
        nn.Sigmoid, nn.Tanh
]:
    MODELS.register_module(module=module)

if digit_version(torch.__version__) >= digit_version('1.7.0'):
    MODELS.register_module(module=nn.SiLU, name='SiLU')
else:

    class SiLU(nn.Module):
        """Sigmoid Weighted Liner Unit."""

        def __init__(self, inplace=False):
            super().__init__()
            self.inplace = inplace

        def forward(self, inputs) -> torch.Tensor:
            if self.inplace:
                return inputs.mul_(torch.sigmoid(inputs))
            else:
                return inputs * torch.sigmoid(inputs)

    MODELS.register_module(module=SiLU, name='SiLU')


@MODELS.register_module(name='Clip')
@MODELS.register_module()
class Clamp(nn.Module):
    """Clamp activation layer.

    This activation function is to clamp the feature map value within
    :math:`[min, max]`. More details can be found in ``torch.clamp()``.

    Args:
        min (Number | optional): Lower-bound of the range to be clamped to.
            Default to -1.
        max (Number | optional): Upper-bound of the range to be clamped to.
            Default to 1.
    """

    def __init__(self, min: float = -1., max: float = 1.):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: Clamped tensor.
        """
        return torch.clamp(x, min=self.min, max=self.max)


class GELU(nn.Module):
    r"""Applies the Gaussian Error Linear Units function:

    .. math::
        \text{GELU}(x) = x * \Phi(x)
    where :math:`\Phi(x)` is the Cumulative Distribution Function for
    Gaussian Distribution.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/GELU.png

    Examples::

        >>> m = nn.GELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.gelu(input)


if (TORCH_VERSION == 'parrots'
        or digit_version(TORCH_VERSION) < digit_version('1.4')):
    MODELS.register_module(module=GELU)
else:
    MODELS.register_module(module=nn.GELU)


def build_activation_layer(cfg: Dict) -> nn.Module:
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    return MODELS.build(cfg)
