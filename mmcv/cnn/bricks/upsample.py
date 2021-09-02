# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
import torch.nn.functional as F

from ..utils import xavier_init
from .registry import UPSAMPLE_LAYERS, build_upsample_layer_from_cfg

UPSAMPLE_LAYERS.register_module('nearest', module=nn.Upsample)
UPSAMPLE_LAYERS.register_module('bilinear', module=nn.Upsample)


@UPSAMPLE_LAYERS.register_module(name='pixel_shuffle')
class PixelShufflePack(nn.Module):
    """Pixel Shuffle upsample layer.

    This module packs `F.pixel_shuffle()` and a nn.Conv2d module together to
    achieve a simple upsampling with pixel shuffle.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of the conv layer to expand the
            channels.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super(PixelShufflePack, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        xavier_init(self.upsample_conv, distribution='uniform')

    def forward(self, x):
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


# Avoid BC-breaking of importing build_upsample_layer.
# It's backwards compatible with the old `build_upsample_layer`.
def build_upsample_layer(cfg, *args, **kwargs):
    """Build upsample layer.

    Args:
        cfg (dict): The upsample layer config, which should contain:

            - type (str): Layer type.
            - scale_factor (int): Upsample ratio, which is not applicable to
                deconv.
            - layer args: Args needed to instantiate a upsample layer.
        args (argument list): Arguments passed to the ``__init__``
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the
            ``__init__`` method of the corresponding conv layer.

    Returns:
        nn.Module: Created upsample layer.
    """

    warnings.warn(
        ImportWarning(
            '``build_upsample_layer(cfg, *args, **kwargs)`` will be '
            'deprecated. Please use '
            '``UPSAMPLE_LAYERS.build(cfg, *args, **kwargs)`` instead.'))

    return build_upsample_layer_from_cfg(cfg, UPSAMPLE_LAYERS, *args, **kwargs)
