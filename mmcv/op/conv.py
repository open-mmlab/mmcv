from .conv_ws import ConvWS2d
from .deform_conv import DeformConv2dPack
from .modulated_deform_conv import ModulatedDeformConv2dPack
from .wrappers import Conv2d

conv_cfg = {
    'Conv': Conv2d,
    'ConvWS': ConvWS2d,
    'DCN': DeformConv2dPack,
    'DCNv2': ModulatedDeformConv2dPack,
    # TODO: octave conv
}


def build_conv_layer(cfg, *args, **kwargs):
    """ Build convolution layer

    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify conv layer type.
            layer args: args needed to instantiate a conv layer.

    Returns:
        layer (nn.Module): created conv layer
    """
    if cfg is None:
        cfg_ = dict(type='Conv')
    else:
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError(f'Unrecognized norm type {layer_type}')
    else:
        conv_layer = conv_cfg[layer_type]

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer
