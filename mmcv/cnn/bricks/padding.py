# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch.nn as nn
from mmengine.registry import MODELS

MODELS.register_module('zero', module=nn.ZeroPad2d)
MODELS.register_module('reflect', module=nn.ReflectionPad2d)
MODELS.register_module('replicate', module=nn.ReplicationPad2d)


def build_padding_layer(cfg: Dict, *args, **kwargs) -> nn.Module:
    """Build padding layer.

    Args:
        cfg (dict): The padding layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a padding layer.

    Returns:
        nn.Module: Created padding layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')

    cfg_ = cfg.copy()
    padding_type = cfg_.pop('type')
    if padding_type not in MODELS:
        raise KeyError(f'Unrecognized padding type {padding_type}.')
    else:
        padding_layer = MODELS.get(padding_type)

    layer = padding_layer(*args, **kwargs, **cfg_)

    return layer
