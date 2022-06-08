# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

import pytest
import torch
from torch import nn

import mmcv
from mmcv.cnn.utils.weight_init import update_init_info
from mmcv.runner import BaseModule, ModuleDict, ModuleList, Sequential
from mmcv.utils import Registry, build_from_cfg

COMPONENTS = Registry('component')
FOOMODELS = Registry('model')


@COMPONENTS.register_module()
class FooConv1d(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.conv1d = nn.Conv1d(4, 1, 4)

    def forward(self, x):
        return self.conv1d(x)


@COMPONENTS.register_module()
class FooConv2d(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.conv2d = nn.Conv2d(3, 1, 3)

    def forward(self, x):
        return self.conv2d(x)


@COMPONENTS.register_module()
class FooLinear(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.linear = nn.Linear(3, 4)

    def forward(self, x):
        return self.linear(x)


@COMPONENTS.register_module()
class FooLinearConv1d(BaseModule):

    def __init__(self, linear=None, conv1d=None, init_cfg=None):
        super().__init__(init_cfg)
        if linear is not None:
            self.linear = build_from_cfg(linear, COMPONENTS)
        if conv1d is not None:
            self.conv1d = build_from_cfg(conv1d, COMPONENTS)

    def forward(self, x):
        x = self.linear(x)
        return self.conv1d(x)


@FOOMODELS.register_module()
class FooModel(BaseModule):

    def __init__(self,
                 component1=None,
                 component2=None,
                 component3=None,
                 component4=None,
                 init_cfg=None) -> None:
        super().__init__(init_cfg)
        if component1 is not None:
            self.component1 = build_from_cfg(component1, COMPONENTS)
        if component2 is not None:
            self.component2 = build_from_cfg(component2, COMPONENTS)
        if component3 is not None:
            self.component3 = build_from_cfg(component3, COMPONENTS)
        if component4 is not None:
            self.component4 = build_from_cfg(component4, COMPONENTS)

        # its type is not BaseModule, it can be initialized
        # with "override" key.
        self.reg = nn.Linear(3, 4)


def test_initilization_info_logger():
    # 'override' has higher priority

    import os

    import torch.nn as nn

    from mmcv.utils.logging import get_logger

    class OverloadInitConv(nn.Conv2d, BaseModule):

        def init_weights(self):
            for p in self.parameters():
                with torch.no_grad():
                    p.fill_(1)

    class CheckLoggerModel(BaseModule):

        def __init__(self, init_cfg=None):
            super().__init__(init_cfg)
            self.conv1 = nn.Conv2d(1, 1, 1, 1)
            self.conv2 = OverloadInitConv(1, 1, 1, 1)
            self.conv3 = nn.Conv2d(1, 1, 1, 1)
            self.fc1 = nn.Linear(1, 1)

    init_cfg = [
        dict(
            type='Normal',
            layer='Conv2d',
            std=0.01,
            override=dict(
                type='Normal', name='conv3', std=0.01, bias_prob=0.01)),
        dict(type='Constant', layer='Linear', val=0., bias=1.)
    ]

    model = CheckLoggerModel(init_cfg=init_cfg)

    train_log = '20210720_132454.log'
    workdir = tempfile.mkdtemp()
    log_file = os.path.join(workdir, train_log)
    # create a logger
    get_logger('init_logger', log_file=log_file)
    assert not hasattr(model, '_params_init_info')
    model.init_weights()
    # assert `_params_init_info` would be deleted after `init_weights`
    assert not hasattr(model, '_params_init_info')
    # assert initialization information has been dumped
    assert os.path.exists(log_file)

    lines = mmcv.list_from_file(log_file)

    # check initialization information is right
    for i, line in enumerate(lines):
        if 'conv1.weight' in line:
            assert 'NormalInit' in lines[i + 1]
        if 'conv2.weight' in line:
            assert 'OverloadInitConv' in lines[i + 1]
        if 'fc1.weight' in line:
            assert 'ConstantInit' in lines[i + 1]

    # test corner case

    class OverloadInitConvFc(nn.Conv2d, BaseModule):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.conv1 = nn.Linear(1, 1)

        def init_weights(self):
            for p in self.parameters():
                with torch.no_grad():
                    p.fill_(1)

    class CheckLoggerModel(BaseModule):

        def __init__(self, init_cfg=None):
            super().__init__(init_cfg)
            self.conv1 = nn.Conv2d(1, 1, 1, 1)
            self.conv2 = OverloadInitConvFc(1, 1, 1, 1)
            self.conv3 = nn.Conv2d(1, 1, 1, 1)
            self.fc1 = nn.Linear(1, 1)

    class TopLevelModule(BaseModule):

        def __init__(self, init_cfg=None, checklog_init_cfg=None):
            super().__init__(init_cfg)
            self.module1 = CheckLoggerModel(checklog_init_cfg)
            self.module2 = OverloadInitConvFc(1, 1, 1, 1)

    checklog_init_cfg = [
        dict(
            type='Normal',
            layer='Conv2d',
            std=0.01,
            override=dict(
                type='Normal', name='conv3', std=0.01, bias_prob=0.01)),
        dict(type='Constant', layer='Linear', val=0., bias=1.)
    ]

    top_level_init_cfg = [
        dict(
            type='Normal',
            layer='Conv2d',
            std=0.01,
            override=dict(
                type='Normal', name='module2', std=0.01, bias_prob=0.01))
    ]

    model = TopLevelModule(
        init_cfg=top_level_init_cfg, checklog_init_cfg=checklog_init_cfg)

    model.module1.init_weights()
    model.module2.init_weights()
    model.init_weights()
    model.module1.init_weights()
    model.module2.init_weights()

    assert not hasattr(model, '_params_init_info')
    model.init_weights()
    # assert `_params_init_info` would be deleted after `init_weights`
    assert not hasattr(model, '_params_init_info')
    # assert initialization information has been dumped
    assert os.path.exists(log_file)

    lines = mmcv.list_from_file(log_file)
    # check initialization information is right
    for i, line in enumerate(lines):
        if 'TopLevelModule' in line and 'init_cfg' not in line:
            # have been set init_flag
            assert 'the same' in line


def test_update_init_info():

    class DummyModel(BaseModule):

        def __init__(self, init_cfg=None):
            super().__init__(init_cfg)
            self.conv1 = nn.Conv2d(1, 1, 1, 1)
            self.conv3 = nn.Conv2d(1, 1, 1, 1)
            self.fc1 = nn.Linear(1, 1)

    model = DummyModel()
    from collections import defaultdict
    model._params_init_info = defaultdict(dict)
    for name, param in model.named_parameters():
        model._params_init_info[param]['init_info'] = 'init'
        model._params_init_info[param]['tmp_mean_value'] = param.data.mean()

    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1)

    update_init_info(model, init_info='fill_1')

    for item in model._params_init_info.values():
        assert item['init_info'] == 'fill_1'
        assert item['tmp_mean_value'] == 1

    # test assert for new parameters
    model.conv1.bias = nn.Parameter(torch.ones_like(model.conv1.bias))
    with pytest.raises(AssertionError):
        update_init_info(model, init_info=' ')


def test_model_weight_init():
    """
    Config
    model (FooModel, Linear: weight=1, bias=2, Conv1d: weight=3, bias=4,
                     Conv2d: weight=5, bias=6)
    ├──component1 (FooConv1d)
    ├──component2 (FooConv2d)
    ├──component3 (FooLinear)
    ├──component4 (FooLinearConv1d)
        ├──linear (FooLinear)
        ├──conv1d (FooConv1d)
    ├──reg (nn.Linear)

    Parameters after initialization
    model (FooModel)
    ├──component1 (FooConv1d, weight=3, bias=4)
    ├──component2 (FooConv2d, weight=5, bias=6)
    ├──component3 (FooLinear, weight=1, bias=2)
    ├──component4 (FooLinearConv1d)
        ├──linear (FooLinear, weight=1, bias=2)
        ├──conv1d (FooConv1d, weight=3, bias=4)
    ├──reg (nn.Linear, weight=1, bias=2)
    """
    model_cfg = dict(
        type='FooModel',
        init_cfg=[
            dict(type='Constant', val=1, bias=2, layer='Linear'),
            dict(type='Constant', val=3, bias=4, layer='Conv1d'),
            dict(type='Constant', val=5, bias=6, layer='Conv2d')
        ],
        component1=dict(type='FooConv1d'),
        component2=dict(type='FooConv2d'),
        component3=dict(type='FooLinear'),
        component4=dict(
            type='FooLinearConv1d',
            linear=dict(type='FooLinear'),
            conv1d=dict(type='FooConv1d')))

    model = build_from_cfg(model_cfg, FOOMODELS)
    model.init_weights()

    assert torch.equal(model.component1.conv1d.weight,
                       torch.full(model.component1.conv1d.weight.shape, 3.0))
    assert torch.equal(model.component1.conv1d.bias,
                       torch.full(model.component1.conv1d.bias.shape, 4.0))
    assert torch.equal(model.component2.conv2d.weight,
                       torch.full(model.component2.conv2d.weight.shape, 5.0))
    assert torch.equal(model.component2.conv2d.bias,
                       torch.full(model.component2.conv2d.bias.shape, 6.0))
    assert torch.equal(model.component3.linear.weight,
                       torch.full(model.component3.linear.weight.shape, 1.0))
    assert torch.equal(model.component3.linear.bias,
                       torch.full(model.component3.linear.bias.shape, 2.0))
    assert torch.equal(
        model.component4.linear.linear.weight,
        torch.full(model.component4.linear.linear.weight.shape, 1.0))
    assert torch.equal(
        model.component4.linear.linear.bias,
        torch.full(model.component4.linear.linear.bias.shape, 2.0))
    assert torch.equal(
        model.component4.conv1d.conv1d.weight,
        torch.full(model.component4.conv1d.conv1d.weight.shape, 3.0))
    assert torch.equal(
        model.component4.conv1d.conv1d.bias,
        torch.full(model.component4.conv1d.conv1d.bias.shape, 4.0))
    assert torch.equal(model.reg.weight, torch.full(model.reg.weight.shape,
                                                    1.0))
    assert torch.equal(model.reg.bias, torch.full(model.reg.bias.shape, 2.0))


def test_nest_components_weight_init():
    """
    Config
    model (FooModel, Linear: weight=1, bias=2, Conv1d: weight=3, bias=4,
                     Conv2d: weight=5, bias=6)
    ├──component1 (FooConv1d, Conv1d: weight=7, bias=8)
    ├──component2 (FooConv2d, Conv2d: weight=9, bias=10)
    ├──component3 (FooLinear)
    ├──component4 (FooLinearConv1d, Linear: weight=11, bias=12)
        ├──linear (FooLinear, Linear: weight=11, bias=12)
        ├──conv1d (FooConv1d)
    ├──reg (nn.Linear, weight=13, bias=14)

    Parameters after initialization
    model (FooModel)
    ├──component1 (FooConv1d, weight=7, bias=8)
    ├──component2 (FooConv2d, weight=9, bias=10)
    ├──component3 (FooLinear, weight=1, bias=2)
    ├──component4 (FooLinearConv1d)
        ├──linear (FooLinear, weight=1, bias=2)
        ├──conv1d (FooConv1d, weight=3, bias=4)
    ├──reg (nn.Linear, weight=13, bias=14)
    """

    model_cfg = dict(
        type='FooModel',
        init_cfg=[
            dict(
                type='Constant',
                val=1,
                bias=2,
                layer='Linear',
                override=dict(type='Constant', name='reg', val=13, bias=14)),
            dict(type='Constant', val=3, bias=4, layer='Conv1d'),
            dict(type='Constant', val=5, bias=6, layer='Conv2d'),
        ],
        component1=dict(
            type='FooConv1d',
            init_cfg=dict(type='Constant', layer='Conv1d', val=7, bias=8)),
        component2=dict(
            type='FooConv2d',
            init_cfg=dict(type='Constant', layer='Conv2d', val=9, bias=10)),
        component3=dict(type='FooLinear'),
        component4=dict(
            type='FooLinearConv1d',
            linear=dict(type='FooLinear'),
            conv1d=dict(type='FooConv1d')))

    model = build_from_cfg(model_cfg, FOOMODELS)
    model.init_weights()

    assert torch.equal(model.component1.conv1d.weight,
                       torch.full(model.component1.conv1d.weight.shape, 7.0))
    assert torch.equal(model.component1.conv1d.bias,
                       torch.full(model.component1.conv1d.bias.shape, 8.0))
    assert torch.equal(model.component2.conv2d.weight,
                       torch.full(model.component2.conv2d.weight.shape, 9.0))
    assert torch.equal(model.component2.conv2d.bias,
                       torch.full(model.component2.conv2d.bias.shape, 10.0))
    assert torch.equal(model.component3.linear.weight,
                       torch.full(model.component3.linear.weight.shape, 1.0))
    assert torch.equal(model.component3.linear.bias,
                       torch.full(model.component3.linear.bias.shape, 2.0))
    assert torch.equal(
        model.component4.linear.linear.weight,
        torch.full(model.component4.linear.linear.weight.shape, 1.0))
    assert torch.equal(
        model.component4.linear.linear.bias,
        torch.full(model.component4.linear.linear.bias.shape, 2.0))
    assert torch.equal(
        model.component4.conv1d.conv1d.weight,
        torch.full(model.component4.conv1d.conv1d.weight.shape, 3.0))
    assert torch.equal(
        model.component4.conv1d.conv1d.bias,
        torch.full(model.component4.conv1d.conv1d.bias.shape, 4.0))
    assert torch.equal(model.reg.weight,
                       torch.full(model.reg.weight.shape, 13.0))
    assert torch.equal(model.reg.bias, torch.full(model.reg.bias.shape, 14.0))


def test_without_layer_weight_init():
    model_cfg = dict(
        type='FooModel',
        init_cfg=[
            dict(type='Constant', val=1, bias=2, layer='Linear'),
            dict(type='Constant', val=3, bias=4, layer='Conv1d'),
            dict(type='Constant', val=5, bias=6, layer='Conv2d')
        ],
        component1=dict(
            type='FooConv1d', init_cfg=dict(type='Constant', val=7, bias=8)),
        component2=dict(type='FooConv2d'),
        component3=dict(type='FooLinear'))
    model = build_from_cfg(model_cfg, FOOMODELS)
    model.init_weights()

    assert torch.equal(model.component1.conv1d.weight,
                       torch.full(model.component1.conv1d.weight.shape, 3.0))
    assert torch.equal(model.component1.conv1d.bias,
                       torch.full(model.component1.conv1d.bias.shape, 4.0))

    # init_cfg in component1 does not have layer key, so it does nothing
    assert torch.equal(model.component2.conv2d.weight,
                       torch.full(model.component2.conv2d.weight.shape, 5.0))
    assert torch.equal(model.component2.conv2d.bias,
                       torch.full(model.component2.conv2d.bias.shape, 6.0))
    assert torch.equal(model.component3.linear.weight,
                       torch.full(model.component3.linear.weight.shape, 1.0))
    assert torch.equal(model.component3.linear.bias,
                       torch.full(model.component3.linear.bias.shape, 2.0))

    assert torch.equal(model.reg.weight, torch.full(model.reg.weight.shape,
                                                    1.0))
    assert torch.equal(model.reg.bias, torch.full(model.reg.bias.shape, 2.0))


def test_override_weight_init():

    # only initialize 'override'
    model_cfg = dict(
        type='FooModel',
        init_cfg=[
            dict(type='Constant', val=10, bias=20, override=dict(name='reg'))
        ],
        component1=dict(type='FooConv1d'),
        component3=dict(type='FooLinear'))
    model = build_from_cfg(model_cfg, FOOMODELS)
    model.init_weights()
    assert torch.equal(model.reg.weight,
                       torch.full(model.reg.weight.shape, 10.0))
    assert torch.equal(model.reg.bias, torch.full(model.reg.bias.shape, 20.0))
    # do not initialize others
    assert not torch.equal(
        model.component1.conv1d.weight,
        torch.full(model.component1.conv1d.weight.shape, 10.0))
    assert not torch.equal(
        model.component1.conv1d.bias,
        torch.full(model.component1.conv1d.bias.shape, 20.0))
    assert not torch.equal(
        model.component3.linear.weight,
        torch.full(model.component3.linear.weight.shape, 10.0))
    assert not torch.equal(
        model.component3.linear.bias,
        torch.full(model.component3.linear.bias.shape, 20.0))

    # 'override' has higher priority
    model_cfg = dict(
        type='FooModel',
        init_cfg=[
            dict(
                type='Constant',
                val=1,
                bias=2,
                override=dict(name='reg', type='Constant', val=30, bias=40))
        ],
        component1=dict(type='FooConv1d'),
        component2=dict(type='FooConv2d'),
        component3=dict(type='FooLinear'))
    model = build_from_cfg(model_cfg, FOOMODELS)
    model.init_weights()

    assert torch.equal(model.reg.weight,
                       torch.full(model.reg.weight.shape, 30.0))
    assert torch.equal(model.reg.bias, torch.full(model.reg.bias.shape, 40.0))


def test_sequential_model_weight_init():
    seq_model_cfg = [
        dict(
            type='FooConv1d',
            init_cfg=dict(type='Constant', layer='Conv1d', val=0., bias=1.)),
        dict(
            type='FooConv2d',
            init_cfg=dict(type='Constant', layer='Conv2d', val=2., bias=3.)),
    ]
    layers = [build_from_cfg(cfg, COMPONENTS) for cfg in seq_model_cfg]
    seq_model = Sequential(*layers)
    seq_model.init_weights()
    assert torch.equal(seq_model[0].conv1d.weight,
                       torch.full(seq_model[0].conv1d.weight.shape, 0.))
    assert torch.equal(seq_model[0].conv1d.bias,
                       torch.full(seq_model[0].conv1d.bias.shape, 1.))
    assert torch.equal(seq_model[1].conv2d.weight,
                       torch.full(seq_model[1].conv2d.weight.shape, 2.))
    assert torch.equal(seq_model[1].conv2d.bias,
                       torch.full(seq_model[1].conv2d.bias.shape, 3.))
    # inner init_cfg has higher priority
    layers = [build_from_cfg(cfg, COMPONENTS) for cfg in seq_model_cfg]
    seq_model = Sequential(
        *layers,
        init_cfg=dict(
            type='Constant', layer=['Conv1d', 'Conv2d'], val=4., bias=5.))
    seq_model.init_weights()
    assert torch.equal(seq_model[0].conv1d.weight,
                       torch.full(seq_model[0].conv1d.weight.shape, 0.))
    assert torch.equal(seq_model[0].conv1d.bias,
                       torch.full(seq_model[0].conv1d.bias.shape, 1.))
    assert torch.equal(seq_model[1].conv2d.weight,
                       torch.full(seq_model[1].conv2d.weight.shape, 2.))
    assert torch.equal(seq_model[1].conv2d.bias,
                       torch.full(seq_model[1].conv2d.bias.shape, 3.))


def test_modulelist_weight_init():
    models_cfg = [
        dict(
            type='FooConv1d',
            init_cfg=dict(type='Constant', layer='Conv1d', val=0., bias=1.)),
        dict(
            type='FooConv2d',
            init_cfg=dict(type='Constant', layer='Conv2d', val=2., bias=3.)),
    ]
    layers = [build_from_cfg(cfg, COMPONENTS) for cfg in models_cfg]
    modellist = ModuleList(layers)
    modellist.init_weights()
    assert torch.equal(modellist[0].conv1d.weight,
                       torch.full(modellist[0].conv1d.weight.shape, 0.))
    assert torch.equal(modellist[0].conv1d.bias,
                       torch.full(modellist[0].conv1d.bias.shape, 1.))
    assert torch.equal(modellist[1].conv2d.weight,
                       torch.full(modellist[1].conv2d.weight.shape, 2.))
    assert torch.equal(modellist[1].conv2d.bias,
                       torch.full(modellist[1].conv2d.bias.shape, 3.))
    # inner init_cfg has higher priority
    layers = [build_from_cfg(cfg, COMPONENTS) for cfg in models_cfg]
    modellist = ModuleList(
        layers,
        init_cfg=dict(
            type='Constant', layer=['Conv1d', 'Conv2d'], val=4., bias=5.))
    modellist.init_weights()
    assert torch.equal(modellist[0].conv1d.weight,
                       torch.full(modellist[0].conv1d.weight.shape, 0.))
    assert torch.equal(modellist[0].conv1d.bias,
                       torch.full(modellist[0].conv1d.bias.shape, 1.))
    assert torch.equal(modellist[1].conv2d.weight,
                       torch.full(modellist[1].conv2d.weight.shape, 2.))
    assert torch.equal(modellist[1].conv2d.bias,
                       torch.full(modellist[1].conv2d.bias.shape, 3.))


def test_moduledict_weight_init():
    models_cfg = dict(
        foo_conv_1d=dict(
            type='FooConv1d',
            init_cfg=dict(type='Constant', layer='Conv1d', val=0., bias=1.)),
        foo_conv_2d=dict(
            type='FooConv2d',
            init_cfg=dict(type='Constant', layer='Conv2d', val=2., bias=3.)),
    )
    layers = {
        name: build_from_cfg(cfg, COMPONENTS)
        for name, cfg in models_cfg.items()
    }
    modeldict = ModuleDict(layers)
    modeldict.init_weights()
    assert torch.equal(
        modeldict['foo_conv_1d'].conv1d.weight,
        torch.full(modeldict['foo_conv_1d'].conv1d.weight.shape, 0.))
    assert torch.equal(
        modeldict['foo_conv_1d'].conv1d.bias,
        torch.full(modeldict['foo_conv_1d'].conv1d.bias.shape, 1.))
    assert torch.equal(
        modeldict['foo_conv_2d'].conv2d.weight,
        torch.full(modeldict['foo_conv_2d'].conv2d.weight.shape, 2.))
    assert torch.equal(
        modeldict['foo_conv_2d'].conv2d.bias,
        torch.full(modeldict['foo_conv_2d'].conv2d.bias.shape, 3.))
    # inner init_cfg has higher priority
    layers = {
        name: build_from_cfg(cfg, COMPONENTS)
        for name, cfg in models_cfg.items()
    }
    modeldict = ModuleDict(
        layers,
        init_cfg=dict(
            type='Constant', layer=['Conv1d', 'Conv2d'], val=4., bias=5.))
    modeldict.init_weights()
    assert torch.equal(
        modeldict['foo_conv_1d'].conv1d.weight,
        torch.full(modeldict['foo_conv_1d'].conv1d.weight.shape, 0.))
    assert torch.equal(
        modeldict['foo_conv_1d'].conv1d.bias,
        torch.full(modeldict['foo_conv_1d'].conv1d.bias.shape, 1.))
    assert torch.equal(
        modeldict['foo_conv_2d'].conv2d.weight,
        torch.full(modeldict['foo_conv_2d'].conv2d.weight.shape, 2.))
    assert torch.equal(
        modeldict['foo_conv_2d'].conv2d.bias,
        torch.full(modeldict['foo_conv_2d'].conv2d.bias.shape, 3.))
