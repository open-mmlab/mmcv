from mmcv.utils import Registry

CONV_LAYERS = Registry('conv layer')
NORM_LAYERS = Registry('norm layer')
ACTIVATION_LAYERS = Registry('activation layer')
PADDING_LAYERS = Registry('padding layer')
UPSAMPLE_LAYERS = Registry('upsample layer')
PLUGIN_LAYERS = Registry('plugin layer')

# Transformer related registry
DROPOUT_LAYERS = Registry('Drop out layers')
POSITIONAL_ENCODING = Registry('Position encoding')
ATTENTION = Registry('Attention')
FEEDFORWARD_NETWORK = Registry('Feed-forward Network')
TRANSFORMER_LAYER = Registry('TransformerLayer')
TRANSFORMER_LAYER_SEQUENCE = Registry('TransformerLayerSequence')
