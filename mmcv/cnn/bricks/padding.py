import torch.nn as nn

pad_cfg = {
    'zero': nn.ZeroPad2d,
    'reflect': nn.ReflectionPad2d,
    'replicate': nn.ReplicationPad2d,
}


def build_padding_layer(cfg, *args, **kwargs):
    """Build padding layer.

    Args:
        cfg (None or dict): The padding layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a padding layer.

    Returns:
        nn.Module: Created padding layer.
    """

    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()
    padding_type = cfg_.pop('type')
    if padding_type not in pad_cfg:
        raise KeyError(f'Unrecognized padding type {padding_type}.')
    else:
        padding_layer = pad_cfg[padding_type]

    layer = padding_layer(*args, **kwargs, **cfg_)

    return layer
