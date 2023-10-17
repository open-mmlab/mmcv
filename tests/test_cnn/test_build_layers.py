# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from importlib import import_module

import numpy as np
import pytest
import torch
import torch.nn as nn
from mmengine.registry import MODELS
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from torch.nn import ReflectionPad2d, Upsample

from mmcv.cnn.bricks import (ContextBlock, ConvModule, ConvTranspose2d,
                             GeneralizedAttention, NonLocal2d,
                             build_activation_layer, build_conv_layer,
                             build_norm_layer, build_padding_layer,
                             build_plugin_layer, build_upsample_layer, is_norm)
from mmcv.cnn.bricks.activation import Clamp
from mmcv.cnn.bricks.norm import infer_abbr as infer_norm_abbr
from mmcv.cnn.bricks.plugin import infer_abbr as infer_plugin_abbr
from mmcv.cnn.bricks.upsample import PixelShufflePack


def test_build_conv_layer():
    with pytest.raises(TypeError):
        # cfg must be a dict
        cfg = 'Conv2d'
        build_conv_layer(cfg)

    with pytest.raises(KeyError):
        # `type` must be in cfg
        cfg = dict(kernel_size=3)
        build_conv_layer(cfg)

    with pytest.raises(KeyError):
        # unsupported conv type
        cfg = dict(type='FancyConv')
        build_conv_layer(cfg)

    kwargs = dict(
        in_channels=4, out_channels=8, kernel_size=3, groups=2, dilation=2)
    cfg = None
    layer = build_conv_layer(cfg, **kwargs)
    assert isinstance(layer, nn.Conv2d)
    assert layer.in_channels == kwargs['in_channels']
    assert layer.out_channels == kwargs['out_channels']
    assert layer.kernel_size == (kwargs['kernel_size'], kwargs['kernel_size'])
    assert layer.groups == kwargs['groups']
    assert layer.dilation == (kwargs['dilation'], kwargs['dilation'])

    cfg = dict(type='Conv')
    layer = build_conv_layer(cfg, **kwargs)
    assert isinstance(layer, nn.Conv2d)
    assert layer.in_channels == kwargs['in_channels']
    assert layer.out_channels == kwargs['out_channels']
    assert layer.kernel_size == (kwargs['kernel_size'], kwargs['kernel_size'])
    assert layer.groups == kwargs['groups']
    assert layer.dilation == (kwargs['dilation'], kwargs['dilation'])

    cfg = dict(type='deconv')
    layer = build_conv_layer(cfg, **kwargs)
    assert isinstance(layer, nn.ConvTranspose2d)
    assert layer.in_channels == kwargs['in_channels']
    assert layer.out_channels == kwargs['out_channels']
    assert layer.kernel_size == (kwargs['kernel_size'], kwargs['kernel_size'])
    assert layer.groups == kwargs['groups']
    assert layer.dilation == (kwargs['dilation'], kwargs['dilation'])

    # sparse convs cannot support the case when groups>1
    kwargs.pop('groups')

    for type_name, module in MODELS.module_dict.items():
        for type_name_ in (type_name, module):
            cfg = dict(type=type_name_)
            # SparseInverseConv2d and SparseInverseConv3d do not have the
            # argument 'dilation'
            if type_name == 'SparseInverseConv2d' or type_name == \
                    'SparseInverseConv3d':
                kwargs.pop('dilation')
            if 'conv' in type_name.lower():
                layer = build_conv_layer(cfg, **kwargs)
                assert isinstance(layer, module)
                assert layer.in_channels == kwargs['in_channels']
                assert layer.out_channels == kwargs['out_channels']
                kwargs['dilation'] = 2  # recover the key


def test_infer_norm_abbr():
    with pytest.raises(TypeError):
        # class_type must be a class
        infer_norm_abbr(0)

    class MyNorm:

        _abbr_ = 'mn'

    assert infer_norm_abbr(MyNorm) == 'mn'

    class FancyBatchNorm:
        pass

    assert infer_norm_abbr(FancyBatchNorm) == 'bn'

    class FancyInstanceNorm:
        pass

    assert infer_norm_abbr(FancyInstanceNorm) == 'in'

    class FancyLayerNorm:
        pass

    assert infer_norm_abbr(FancyLayerNorm) == 'ln'

    class FancyGroupNorm:
        pass

    assert infer_norm_abbr(FancyGroupNorm) == 'gn'

    class FancyNorm:
        pass

    assert infer_norm_abbr(FancyNorm) == 'norm_layer'


def test_build_norm_layer():
    with pytest.raises(TypeError):
        # cfg must be a dict
        cfg = 'BN'
        build_norm_layer(cfg, 3)

    with pytest.raises(KeyError):
        # `type` must be in cfg
        cfg = dict()
        build_norm_layer(cfg, 3)

    with pytest.raises(KeyError):
        # unsupported norm type
        cfg = dict(type='FancyNorm')
        build_norm_layer(cfg, 3)

    with pytest.raises(AssertionError):
        # postfix must be int or str
        cfg = dict(type='BN')
        build_norm_layer(cfg, 3, postfix=[1, 2])

    with pytest.raises(AssertionError):
        # `num_groups` must be in cfg when using 'GN'
        cfg = dict(type='GN')
        build_norm_layer(cfg, 3)

    # test each type of norm layer in norm_cfg
    abbr_mapping = {
        'BN': 'bn',
        'BN1d': 'bn',
        'BN2d': 'bn',
        'BN3d': 'bn',
        'SyncBN': 'bn',
        'GN': 'gn',
        'LN': 'ln',
        'IN': 'in',
        'IN1d': 'in',
        'IN2d': 'in',
        'IN3d': 'in',
    }
    for type_name, module in MODELS.module_dict.items():
        if type_name not in abbr_mapping:
            continue
        if type_name == 'MMSyncBN':  # skip MMSyncBN
            continue
        for postfix in ['_test', 1]:
            for type_name_ in (type_name, module):
                cfg = dict(type=type_name_)
                if type_name == 'GN':
                    cfg['num_groups'] = 3
                name, layer = build_norm_layer(cfg, 3, postfix=postfix)
                assert name == abbr_mapping[type_name] + str(postfix)
                assert isinstance(layer, module)
                if type_name == 'GN':
                    assert layer.num_channels == 3
                    assert layer.num_groups == cfg['num_groups']
                elif type_name != 'LN':
                    assert layer.num_features == 3


def test_build_activation_layer():
    act_names = [
        'ReLU', 'LeakyReLU', 'PReLU', 'RReLU', 'ReLU6', 'ELU', 'Sigmoid',
        'Tanh'
    ]

    for module_name in ['activation', 'hsigmoid', 'hswish', 'swish']:
        act_module = import_module(f'mmcv.cnn.bricks.{module_name}')
        for key, value in act_module.__dict__.items():
            if inspect.isclass(value) and issubclass(value, nn.Module):
                act_names.append(key)

    with pytest.raises(TypeError):
        # cfg must be a dict
        cfg = 'ReLU'
        build_activation_layer(cfg)

    with pytest.raises(KeyError):
        # `type` must be in cfg
        cfg = dict()
        build_activation_layer(cfg)

    with pytest.raises(KeyError):
        # unsupported activation type
        cfg = dict(type='FancyReLU')
        build_activation_layer(cfg)

    # test each type of activation layer in activation_cfg
    for type_name, module in MODELS.module_dict.items():
        if type_name in act_names:
            cfg['type'] = type_name
            layer = build_activation_layer(cfg)
            assert isinstance(layer, module)

    # sanity check for Clamp
    for type_name in ('Clamp', Clamp):
        act = build_activation_layer(dict(type='Clamp'))
        x = torch.randn(10) * 1000
        y = act(x)
        assert np.logical_and((y >= -1).numpy(), (y <= 1).numpy()).all()

    act = build_activation_layer(dict(type='Clip', min=0))
    y = act(x)
    assert np.logical_and((y >= 0).numpy(), (y <= 1).numpy()).all()
    act = build_activation_layer(dict(type='Clamp', max=0))
    y = act(x)
    assert np.logical_and((y >= -1).numpy(), (y <= 0).numpy()).all()


def test_build_padding_layer():
    pad_names = ['zero', 'reflect', 'replicate']
    for module_name in ['padding']:
        pad_module = import_module(f'mmcv.cnn.bricks.{module_name}')
        for key, value in pad_module.__dict__.items():
            if inspect.isclass(value) and issubclass(value, nn.Module):
                pad_names.append(key)

    with pytest.raises(TypeError):
        # cfg must be a dict
        cfg = 'reflect'
        build_padding_layer(cfg)

    with pytest.raises(KeyError):
        # `type` must be in cfg
        cfg = dict()
        build_padding_layer(cfg)

    with pytest.raises(KeyError):
        # unsupported activation type
        cfg = dict(type='FancyPad')
        build_padding_layer(cfg)

    for type_name, module in MODELS.module_dict.items():
        if type_name in pad_names:
            cfg['type'] = type_name
            layer = build_padding_layer(cfg, 2)
            assert isinstance(layer, module)
    for type_name in (ReflectionPad2d, 'reflect'):
        input_x = torch.randn(1, 2, 5, 5)
        cfg = dict(type=type_name)
        padding_layer = build_padding_layer(cfg, 2)
        res = padding_layer(input_x)
        assert res.shape == (1, 2, 9, 9)


def test_upsample_layer():
    with pytest.raises(TypeError):
        # cfg must be a dict
        cfg = 'bilinear'
        build_upsample_layer(cfg)

    with pytest.raises(KeyError):
        # `type` must be in cfg
        cfg = dict()
        build_upsample_layer(cfg)

    with pytest.raises(KeyError):
        # unsupported activation type
        cfg = dict(type='FancyUpsample')
        build_upsample_layer(cfg)

    for type_name in ['nearest', 'bilinear']:
        cfg['type'] = type_name
        layer = build_upsample_layer(cfg)
        assert isinstance(layer, nn.Upsample)
        assert layer.mode == type_name

    cfg = dict()
    cfg['type'] = Upsample
    layer_from_cls = build_upsample_layer(cfg)
    assert isinstance(layer_from_cls, nn.Upsample)
    assert layer_from_cls.mode == 'nearest'

    cfg = dict(
        type='deconv', in_channels=3, out_channels=3, kernel_size=3, stride=2)
    layer = build_upsample_layer(cfg)
    assert isinstance(layer, nn.ConvTranspose2d)

    for type_name in ('deconv', ConvTranspose2d):
        cfg = dict(type=ConvTranspose2d)
        kwargs = dict(in_channels=3, out_channels=3, kernel_size=3, stride=2)
        layer = build_upsample_layer(cfg, **kwargs)
        assert isinstance(layer, nn.ConvTranspose2d)
        assert layer.in_channels == kwargs['in_channels']
        assert layer.out_channels == kwargs['out_channels']
        assert layer.kernel_size == (kwargs['kernel_size'],
                                     kwargs['kernel_size'])
        assert layer.stride == (kwargs['stride'], kwargs['stride'])

        layer = build_upsample_layer(cfg, 3, 3, 3, 2)
        assert isinstance(layer, nn.ConvTranspose2d)
        assert layer.in_channels == kwargs['in_channels']
        assert layer.out_channels == kwargs['out_channels']
        assert layer.kernel_size == (kwargs['kernel_size'],
                                     kwargs['kernel_size'])
        assert layer.stride == (kwargs['stride'], kwargs['stride'])

    for type_name in ('pixel_shuffle', PixelShufflePack):
        cfg = dict(
            type=type_name,
            in_channels=3,
            out_channels=3,
            scale_factor=2,
            upsample_kernel=3)
        layer = build_upsample_layer(cfg)

        assert isinstance(layer, PixelShufflePack)
        assert layer.scale_factor == 2
        assert layer.upsample_kernel == 3


def test_pixel_shuffle_pack():
    x_in = torch.rand(2, 3, 10, 10)
    pixel_shuffle = PixelShufflePack(3, 3, scale_factor=2, upsample_kernel=3)
    assert pixel_shuffle.upsample_conv.kernel_size == (3, 3)
    x_out = pixel_shuffle(x_in)
    assert x_out.shape == (2, 3, 20, 20)


def test_is_norm():
    norm_set1 = [
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm1d,
        nn.InstanceNorm2d, nn.InstanceNorm3d, nn.LayerNorm
    ]
    norm_set2 = [nn.GroupNorm]
    for norm_type in norm_set1:
        layer = norm_type(3)
        assert is_norm(layer)
        assert not is_norm(layer, exclude=(norm_type, ))
    for norm_type in norm_set2:
        layer = norm_type(3, 6)
        assert is_norm(layer)
        assert not is_norm(layer, exclude=(norm_type, ))

    class MyNorm(nn.BatchNorm2d):
        pass

    layer = MyNorm(3)
    assert is_norm(layer)
    assert not is_norm(layer, exclude=_BatchNorm)
    assert not is_norm(layer, exclude=(_BatchNorm, ))

    layer = nn.Conv2d(3, 8, 1)
    assert not is_norm(layer)

    with pytest.raises(TypeError):
        layer = nn.BatchNorm1d(3)
        is_norm(layer, exclude='BN')

    with pytest.raises(TypeError):
        layer = nn.BatchNorm1d(3)
        is_norm(layer, exclude=('BN', ))


def test_infer_plugin_abbr():
    with pytest.raises(TypeError):
        # class_type must be a class
        infer_plugin_abbr(0)

    class MyPlugin:

        _abbr_ = 'mp'

    assert infer_plugin_abbr(MyPlugin) == 'mp'

    class FancyPlugin:
        pass

    assert infer_plugin_abbr(FancyPlugin) == 'fancy_plugin'


def test_build_plugin_layer():
    with pytest.raises(TypeError):
        # cfg must be a dict
        cfg = 'Plugin'
        build_plugin_layer(cfg)

    with pytest.raises(KeyError):
        # `type` must be in cfg
        cfg = dict()
        build_plugin_layer(cfg)

    with pytest.raises(KeyError):
        # unsupported plugin type
        cfg = dict(type='FancyPlugin')
        build_plugin_layer(cfg)

    with pytest.raises(AssertionError):
        # postfix must be int or str
        cfg = dict(type='ConvModule')
        build_plugin_layer(cfg, postfix=[1, 2])

    # test ContextBlock
    for type_name in ('ContextBlock', ContextBlock):
        for postfix in ['', '_test', 1]:
            cfg = dict(type=type_name)
            name, layer = build_plugin_layer(
                cfg, postfix=postfix, in_channels=16, ratio=1. / 4)
            assert name == 'context_block' + str(postfix)
            assert isinstance(layer, MODELS.module_dict['ContextBlock'])

    # test GeneralizedAttention
    for type_name in ('GeneralizedAttention', GeneralizedAttention):
        for postfix in ['', '_test', 1]:
            cfg = dict(type=type_name)
            name, layer = build_plugin_layer(
                cfg, postfix=postfix, in_channels=16)
            assert name == 'gen_attention_block' + str(postfix)
            assert isinstance(layer,
                              MODELS.module_dict['GeneralizedAttention'])

    # test NonLocal2d
    for type_name in ('NonLocal2d', NonLocal2d):
        for postfix in ['', '_test', 1]:
            cfg = dict(type='NonLocal2d')
            name, layer = build_plugin_layer(
                cfg, postfix=postfix, in_channels=16)
            assert name == 'nonlocal_block' + str(postfix)
            assert isinstance(layer, MODELS.module_dict['NonLocal2d'])

    # test ConvModule
    for postfix in ['', '_test', 1]:
        for type_name in ('ConvModule', ConvModule):
            cfg = dict(type=type_name)
            name, layer = build_plugin_layer(
                cfg,
                postfix=postfix,
                in_channels=16,
                out_channels=4,
                kernel_size=3)
            assert name == 'conv_block' + str(postfix)
            assert isinstance(layer, MODELS.module_dict['ConvModule'])
