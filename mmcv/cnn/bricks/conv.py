# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

from mmengine.registry import MODELS
from torch import nn

MODELS.register_module('Conv1d', module=nn.Conv1d)
MODELS.register_module('Conv2d', module=nn.Conv2d)
MODELS.register_module('Conv3d', module=nn.Conv3d)
MODELS.register_module('Conv', module=nn.Conv2d)


def build_conv_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in MODELS:
        raise KeyError(f'Unrecognized layer type {layer_type}')
    else:
        conv_layer = MODELS.get(layer_type)

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer
