# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn.alexnet import AlexNet

# yapf: disable
from mmcv.cnn.bricks import (
                     ContextBlock,
                     Conv2d,
                     Conv3d,
                     ConvAWS2d,
                     ConvModule,
                     ConvTranspose2d,
                     ConvTranspose3d,
                     ConvWS2d,
                     DepthwiseSeparableConvModule,
                     GeneralizedAttention,
                     HSigmoid,
                     HSwish,
                     Linear,
                     MaxPool2d,
                     MaxPool3d,
                     NonLocal1d,
                     NonLocal2d,
                     NonLocal3d,
                     Scale,
                     Swish,
                     build_activation_layer,
                     build_conv_layer,
                     build_norm_layer,
                     build_padding_layer,
                     build_plugin_layer,
                     build_upsample_layer,
                     conv_ws_2d,
                     is_norm,
)

# yapf: enable
from mmcv.cnn.resnet import ResNet, make_res_layer
from mmcv.cnn.rfsearch import Conv2dRFSearchOp, RFSearchHook
from mmcv.cnn.utils import fuse_conv_bn, get_model_complexity_info
from mmcv.cnn.vgg import VGG, make_vgg_layer

__all__ = [
                     'VGG',
                     'AlexNet',
                     'ContextBlock',
                     'Conv2d',
                     'Conv2dRFSearchOp',
                     'Conv3d',
                     'ConvAWS2d',
                     'ConvModule',
                     'ConvTranspose2d',
                     'ConvTranspose3d',
                     'ConvWS2d',
                     'DepthwiseSeparableConvModule',
                     'GeneralizedAttention',
                     'HSigmoid',
                     'HSwish',
                     'Linear',
                     'MaxPool2d',
                     'MaxPool3d',
                     'NonLocal1d',
                     'NonLocal2d',
                     'NonLocal3d',
                     'RFSearchHook',
                     'ResNet',
                     'Scale',
                     'Swish',
                     'build_activation_layer',
                     'build_conv_layer',
                     'build_norm_layer',
                     'build_padding_layer',
                     'build_plugin_layer',
                     'build_upsample_layer',
                     'conv_ws_2d',
                     'fuse_conv_bn',
                     'get_model_complexity_info',
                     'is_norm',
                     'make_res_layer',
                     'make_vgg_layer'
]
