# Copyright (c) OpenMMLab. All rights reserved.
from .alexnet import AlexNet
# yapf: disable
from .bricks import (ContextBlock, Conv2d, Conv3d, ConvAWS2d, ConvModule,
                     ConvTranspose2d, ConvTranspose3d, ConvWS2d,
                     DepthwiseSeparableConvModule, GeneralizedAttention,
                     HSigmoid, HSwish, Linear, MaxPool2d, MaxPool3d,
                     NonLocal1d, NonLocal2d, NonLocal3d, Scale, Swish,
                     build_activation_layer, build_conv_layer,
                     build_norm_layer, build_padding_layer, build_plugin_layer,
                     build_upsample_layer, conv_ws_2d, is_norm)
# yapf: enable
from .resnet import ResNet, make_res_layer
from .rfsearch import Conv2dRFSearchOp, RFSearchHook
from .utils import fuse_conv_bn, get_model_complexity_info
from .vgg import VGG, make_vgg_layer

__all__ = [
    'AlexNet', 'VGG', 'make_vgg_layer', 'ResNet', 'make_res_layer',
    'ConvModule', 'build_activation_layer', 'build_conv_layer',
    'build_norm_layer', 'build_padding_layer', 'build_upsample_layer',
    'build_plugin_layer', 'is_norm', 'NonLocal1d', 'NonLocal2d', 'NonLocal3d',
    'ContextBlock', 'HSigmoid', 'Swish', 'HSwish', 'GeneralizedAttention',
    'Scale', 'conv_ws_2d', 'ConvAWS2d', 'ConvWS2d',
    'DepthwiseSeparableConvModule', 'Linear', 'Conv2d', 'ConvTranspose2d',
    'MaxPool2d', 'ConvTranspose3d', 'MaxPool3d', 'Conv3d', 'fuse_conv_bn',
    'get_model_complexity_info', 'Conv2dRFSearchOp', 'RFSearchHook'
]
