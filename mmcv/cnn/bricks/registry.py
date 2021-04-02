from mmcv.utils import Registry

CONV_LAYERS = Registry('conv layer')
NORM_LAYERS = Registry('norm layer')
ACTIVATION_LAYERS = Registry('activation layer')
PADDING_LAYERS = Registry('padding layer')
UPSAMPLE_LAYERS = Registry('upsample layer')
PLUGIN_LAYERS = Registry('plugin layer')

POSITIONAL_ENCODING = Registry('Position encoding')
ATTENTION = Registry('Attention')
TRANSFORMER_LAYER = Registry('TransformerLayer')
TRANSFORMER_LAYER_SEQUENCE = Registry('TransformerLayerSequence')
