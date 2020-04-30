from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from mmcv.cnn.bricks import ConvModule


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
    assert not hasattr(conv, 'norm')
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    # conv
    conv = ConvModule(3, 8, 2, act_cfg=None)
    assert not conv.with_norm
    assert not hasattr(conv, 'norm')
    assert not conv.with_activation
    assert not hasattr(conv, 'activate')
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

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

    # bias: True, with norm
    with pytest.warns(UserWarning) as record:
        ConvModule(3, 8, 2, bias=True, norm_cfg=dict(type='BN'))
    assert len(record) == 1
    assert record[0].message.args[
        0] == 'ConvModule has norm and bias at the same time'


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
