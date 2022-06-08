# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from mmcv.cnn.bricks import CONV_LAYERS, ConvModule, HSigmoid, HSwish
from mmcv.utils import TORCH_VERSION, digit_version


@CONV_LAYERS.register_module()
class ExampleConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 norm_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.norm_cfg = norm_cfg
        self.output_padding = (0, 0, 0)
        self.transposed = False

        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.init_weights()

    def forward(self, x):
        x = self.conv0(x)
        return x

    def init_weights(self):
        nn.init.constant_(self.conv0.weight, 0)


def test_conv_module():
    with pytest.raises(AssertionError):
        # conv_cfg must be a dict or None
        conv_cfg = 'conv'
        ConvModule(3, 8, 2, conv_cfg=conv_cfg)

    with pytest.raises(AssertionError):
        # norm_cfg must be a dict or None
        norm_cfg = 'norm'
        ConvModule(3, 8, 2, norm_cfg=norm_cfg)

    with pytest.raises(KeyError):
        # softmax is not supported
        act_cfg = dict(type='softmax')
        ConvModule(3, 8, 2, act_cfg=act_cfg)

    # conv + norm + act
    conv = ConvModule(3, 8, 2, norm_cfg=dict(type='BN'))
    assert conv.with_activation
    assert hasattr(conv, 'activate')
    assert conv.with_norm
    assert hasattr(conv, 'norm')
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    # conv + act
    conv = ConvModule(3, 8, 2)
    assert conv.with_activation
    assert hasattr(conv, 'activate')
    assert not conv.with_norm
    assert conv.norm is None
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    # conv
    conv = ConvModule(3, 8, 2, act_cfg=None)
    assert not conv.with_norm
    assert conv.norm is None
    assert not conv.with_activation
    assert not hasattr(conv, 'activate')
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    # conv with its own `init_weights` method
    conv_module = ConvModule(
        3, 8, 2, conv_cfg=dict(type='ExampleConv'), act_cfg=None)
    assert torch.equal(conv_module.conv.conv0.weight, torch.zeros(8, 3, 2, 2))

    # with_spectral_norm=True
    conv = ConvModule(3, 8, 3, padding=1, with_spectral_norm=True)
    assert hasattr(conv.conv, 'weight_orig')
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # padding_mode='reflect'
    conv = ConvModule(3, 8, 3, padding=1, padding_mode='reflect')
    assert isinstance(conv.padding_layer, nn.ReflectionPad2d)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # non-existing padding mode
    with pytest.raises(KeyError):
        conv = ConvModule(3, 8, 3, padding=1, padding_mode='non_exists')

    # leaky relu
    conv = ConvModule(3, 8, 3, padding=1, act_cfg=dict(type='LeakyReLU'))
    assert isinstance(conv.activate, nn.LeakyReLU)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # tanh
    conv = ConvModule(3, 8, 3, padding=1, act_cfg=dict(type='Tanh'))
    assert isinstance(conv.activate, nn.Tanh)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # Sigmoid
    conv = ConvModule(3, 8, 3, padding=1, act_cfg=dict(type='Sigmoid'))
    assert isinstance(conv.activate, nn.Sigmoid)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # PReLU
    conv = ConvModule(3, 8, 3, padding=1, act_cfg=dict(type='PReLU'))
    assert isinstance(conv.activate, nn.PReLU)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # HSwish
    conv = ConvModule(3, 8, 3, padding=1, act_cfg=dict(type='HSwish'))
    if (TORCH_VERSION == 'parrots'
            or digit_version(TORCH_VERSION) < digit_version('1.7')):
        assert isinstance(conv.activate, HSwish)
    else:
        assert isinstance(conv.activate, nn.Hardswish)

    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # HSigmoid
    conv = ConvModule(3, 8, 3, padding=1, act_cfg=dict(type='HSigmoid'))
    assert isinstance(conv.activate, HSigmoid)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)


def test_bias():
    # bias: auto, without norm
    conv = ConvModule(3, 8, 2)
    assert conv.conv.bias is not None

    # bias: auto, with norm
    conv = ConvModule(3, 8, 2, norm_cfg=dict(type='BN'))
    assert conv.conv.bias is None

    # bias: False, without norm
    conv = ConvModule(3, 8, 2, bias=False)
    assert conv.conv.bias is None

    # bias: True, with batch norm
    with pytest.warns(UserWarning) as record:
        ConvModule(3, 8, 2, bias=True, norm_cfg=dict(type='BN'))
    assert len(record) == 1
    assert record[0].message.args[
        0] == 'Unnecessary conv bias before batch/instance norm'

    # bias: True, with instance norm
    with pytest.warns(UserWarning) as record:
        ConvModule(3, 8, 2, bias=True, norm_cfg=dict(type='IN'))
    assert len(record) == 1
    assert record[0].message.args[
        0] == 'Unnecessary conv bias before batch/instance norm'

    # bias: True, with other norm
    with pytest.warns(UserWarning) as record:
        norm_cfg = dict(type='GN', num_groups=1)
        ConvModule(3, 8, 2, bias=True, norm_cfg=norm_cfg)
        warnings.warn('No warnings')
    assert len(record) == 1
    assert record[0].message.args[0] == 'No warnings'


def conv_forward(self, x):
    return x + '_conv'


def bn_forward(self, x):
    return x + '_bn'


def relu_forward(self, x):
    return x + '_relu'


@patch('torch.nn.ReLU.forward', relu_forward)
@patch('torch.nn.BatchNorm2d.forward', bn_forward)
@patch('torch.nn.Conv2d.forward', conv_forward)
def test_order():

    with pytest.raises(AssertionError):
        # order must be a tuple
        order = ['conv', 'norm', 'act']
        ConvModule(3, 8, 2, order=order)

    with pytest.raises(AssertionError):
        # length of order must be 3
        order = ('conv', 'norm')
        ConvModule(3, 8, 2, order=order)

    with pytest.raises(AssertionError):
        # order must be an order of 'conv', 'norm', 'act'
        order = ('conv', 'norm', 'norm')
        ConvModule(3, 8, 2, order=order)

    with pytest.raises(AssertionError):
        # order must be an order of 'conv', 'norm', 'act'
        order = ('conv', 'norm', 'something')
        ConvModule(3, 8, 2, order=order)

    # ('conv', 'norm', 'act')
    conv = ConvModule(3, 8, 2, norm_cfg=dict(type='BN'))
    out = conv('input')
    assert out == 'input_conv_bn_relu'

    # ('norm', 'conv', 'act')
    conv = ConvModule(
        3, 8, 2, norm_cfg=dict(type='BN'), order=('norm', 'conv', 'act'))
    out = conv('input')
    assert out == 'input_bn_conv_relu'

    # ('conv', 'norm', 'act'), activate=False
    conv = ConvModule(3, 8, 2, norm_cfg=dict(type='BN'))
    out = conv('input', activate=False)
    assert out == 'input_conv_bn'

    # ('conv', 'norm', 'act'), activate=False
    conv = ConvModule(3, 8, 2, norm_cfg=dict(type='BN'))
    out = conv('input', norm=False)
    assert out == 'input_conv_relu'
