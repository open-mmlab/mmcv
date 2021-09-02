# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmcv.utils import Registry


def build_upsample_layer_from_cfg(cfg, registry, *args, **kwargs):
    """Build upsample layer.

    Args:
        cfg (dict): The upsample layer config, which should contain:

            - type (str): Layer type.
            - scale_factor (int): Upsample ratio, which is not applicable to
                deconv.
            - layer args: Args needed to instantiate a upsample layer.
        registry (:obj:`Registry`): A registry the module belongs to.
        args (argument list): Arguments passed to the ``__init__``
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the
            ``__init__`` method of the corresponding conv layer.

    Returns:
        nn.Module: Created upsample layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(
            f'the cfg dict must contain the key "type", but got {cfg}')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in registry:
        raise KeyError(f'Unrecognized upsample type {layer_type}')
    else:
        upsample = registry.get(layer_type)

    if upsample is nn.Upsample:
        cfg_['mode'] = layer_type
    layer = upsample(*args, **kwargs, **cfg_)
    return layer


CONV_LAYERS = Registry('conv layer')
NORM_LAYERS = Registry('norm layer')
ACTIVATION_LAYERS = Registry('activation layer')
PADDING_LAYERS = Registry('padding layer')
PLUGIN_LAYERS = Registry('plugin layer')
UPSAMPLE_LAYERS = Registry(
    'upsample layer', build_func=build_upsample_layer_from_cfg)
DROPOUT_LAYERS = Registry('drop out layers')
POSITIONAL_ENCODING = Registry('position encoding')
ATTENTION = Registry('attention')
FEEDFORWARD_NETWORK = Registry('feed-forward Network')
TRANSFORMER_LAYER = Registry('transformerLayer')
TRANSFORMER_LAYER_SEQUENCE = Registry('transformer-layers sequence')
