import torch.nn as nn

NORM_CONFIG = {
    # layer_type: (abbreviation, module)
    'BN': ('bn', nn.BatchNorm2d),
    'BN1d': ('bn', nn.BatchNorm1d),
    'BN2d': ('bn', nn.BatchNorm2d),
    'BN3d': ('bn', nn.BatchNorm3d),
    'SyncBN': ('bn', nn.SyncBatchNorm),
    'GN': ('gn', nn.GroupNorm),
}


def build_norm_layer(cfg, num_features, postfix=''):
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]:
            name (str): The layer name consisting of abbreviation and postfix,
                e.g., bn1, gn.
            layer (nn.Module): Created norm layer.
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in NORM_CONFIG:
        raise KeyError(f'Unrecognized norm type {layer_type}')
    else:
        abbr, norm_layer = NORM_CONFIG[layer_type]

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN':
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer
