# Copyright (c) Open-MMLab. All rights reserved.
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
from torch import nn

from mmcv.cnn import (ConstantInit, KaimingInit, NormalInit, PretrainedInit,
                      UniformInit, XavierInit, bias_init_with_prob,
                      caffe2_xavier_init, constant_init, initialize,
                      kaiming_init, normal_init, uniform_init, xavier_init)


def test_constant_init():
    conv_module = nn.Conv2d(3, 16, 3)
    constant_init(conv_module, 0.1)
    assert conv_module.weight.allclose(
        torch.full_like(conv_module.weight, 0.1))
    assert conv_module.bias.allclose(torch.zeros_like(conv_module.bias))
    conv_module_no_bias = nn.Conv2d(3, 16, 3, bias=False)
    constant_init(conv_module_no_bias, 0.1)
    assert conv_module.weight.allclose(
        torch.full_like(conv_module.weight, 0.1))


def test_xavier_init():
    conv_module = nn.Conv2d(3, 16, 3)
    xavier_init(conv_module, bias=0.1)
    assert conv_module.bias.allclose(torch.full_like(conv_module.bias, 0.1))
    xavier_init(conv_module, distribution='uniform')
    # TODO: sanity check of weight distribution, e.g. mean, std
    with pytest.raises(AssertionError):
        xavier_init(conv_module, distribution='student-t')
    conv_module_no_bias = nn.Conv2d(3, 16, 3, bias=False)
    xavier_init(conv_module_no_bias)


def test_normal_init():
    conv_module = nn.Conv2d(3, 16, 3)
    normal_init(conv_module, bias=0.1)
    # TODO: sanity check of weight distribution, e.g. mean, std
    assert conv_module.bias.allclose(torch.full_like(conv_module.bias, 0.1))
    conv_module_no_bias = nn.Conv2d(3, 16, 3, bias=False)
    normal_init(conv_module_no_bias)
    # TODO: sanity check distribution, e.g. mean, std


def test_uniform_init():
    conv_module = nn.Conv2d(3, 16, 3)
    uniform_init(conv_module, bias=0.1)
    # TODO: sanity check of weight distribution, e.g. mean, std
    assert conv_module.bias.allclose(torch.full_like(conv_module.bias, 0.1))
    conv_module_no_bias = nn.Conv2d(3, 16, 3, bias=False)
    uniform_init(conv_module_no_bias)


def test_kaiming_init():
    conv_module = nn.Conv2d(3, 16, 3)
    kaiming_init(conv_module, bias=0.1)
    # TODO: sanity check of weight distribution, e.g. mean, std
    assert conv_module.bias.allclose(torch.full_like(conv_module.bias, 0.1))
    kaiming_init(conv_module, distribution='uniform')
    with pytest.raises(AssertionError):
        kaiming_init(conv_module, distribution='student-t')
    conv_module_no_bias = nn.Conv2d(3, 16, 3, bias=False)
    kaiming_init(conv_module_no_bias)


def test_caffe_xavier_init():
    conv_module = nn.Conv2d(3, 16, 3)
    caffe2_xavier_init(conv_module)


def test_bias_init_with_prob():
    conv_module = nn.Conv2d(3, 16, 3)
    prior_prob = 0.1
    normal_init(conv_module, bias=bias_init_with_prob(0.1))
    # TODO: sanity check of weight distribution, e.g. mean, std
    bias = float(-np.log((1 - prior_prob) / prior_prob))
    assert conv_module.bias.allclose(torch.full_like(conv_module.bias, bias))


def test_constaninit():
    """test ConstantInit class."""
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Linear(1, 2))
    func = ConstantInit(val=1, bias=2, layer='Conv2d')
    func(model)
    assert torch.equal(model[0].weight, torch.full(model[0].weight.shape, 1.))
    assert torch.equal(model[0].bias, torch.full(model[0].bias.shape, 2.))

    assert not torch.equal(model[2].weight,
                           torch.full(model[2].weight.shape, 1.))
    assert not torch.equal(model[2].bias, torch.full(model[2].bias.shape, 2.))

    func = ConstantInit(val=3, bias_prob=0.01, layer='Linear')
    func(model)
    res = bias_init_with_prob(0.01)

    assert torch.equal(model[0].weight, torch.full(model[0].weight.shape, 1.))
    assert torch.equal(model[2].weight, torch.full(model[2].weight.shape, 3.))
    assert torch.equal(model[0].bias, torch.full(model[0].bias.shape, 2.))
    assert torch.equal(model[2].bias, torch.full(model[2].bias.shape, res))

    func = ConstantInit(val=4, bias=5)
    func(model)
    assert torch.equal(model[0].weight, torch.full(model[0].weight.shape, 4.))
    assert torch.equal(model[2].weight, torch.full(model[2].weight.shape, 4.))
    assert torch.equal(model[0].bias, torch.full(model[0].bias.shape, 5.))
    assert torch.equal(model[2].bias, torch.full(model[2].bias.shape, 5.))

    # test bias input type
    with pytest.raises(TypeError):
        func = ConstantInit(val=1, bias='1')
    # test bias_prob type
    with pytest.raises(TypeError):
        func = ConstantInit(val=1, bias_prob='1')
    # test layer input type
    with pytest.raises(TypeError):
        func = ConstantInit(val=1, layer=1)


def test_xavierinit():
    """test XavierInit class."""
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Linear(1, 2))
    func = XavierInit(bias=0.1, layer='Conv2d')
    func(model)
    assert model[0].bias.allclose(torch.full_like(model[2].bias, 0.1))
    assert not model[2].bias.allclose(torch.full_like(model[0].bias, 0.1))

    constant_func = ConstantInit(val=0, bias=0)
    func = XavierInit(gain=100, bias_prob=0.01)
    model.apply(constant_func)
    assert torch.equal(model[0].weight, torch.full(model[0].weight.shape, 0.))
    assert torch.equal(model[2].weight, torch.full(model[2].weight.shape, 0.))
    assert torch.equal(model[0].bias, torch.full(model[0].bias.shape, 0.))
    assert torch.equal(model[2].bias, torch.full(model[2].bias.shape, 0.))

    res = bias_init_with_prob(0.01)
    func(model)
    assert not torch.equal(model[0].weight,
                           torch.full(model[0].weight.shape, 0.))
    assert not torch.equal(model[2].weight,
                           torch.full(model[2].weight.shape, 0.))
    assert torch.equal(model[0].bias, torch.full(model[0].bias.shape, res))
    assert torch.equal(model[2].bias, torch.full(model[2].bias.shape, res))

    # test bias input type
    with pytest.raises(TypeError):
        func = XavierInit(bias='0.1', layer='Conv2d')
    # test layer inpur type
    with pytest.raises(TypeError):
        func = XavierInit(bias=0.1, layer=1)


def test_normalinit():
    """test Normalinit class."""
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Linear(1, 2))

    func = NormalInit(mean=100, std=1e-5, bias=200)
    func(model)
    assert model[0].weight.allclose(torch.tensor(100.))
    assert model[2].weight.allclose(torch.tensor(100.))
    assert model[0].bias.allclose(torch.tensor(200.))
    assert model[2].bias.allclose(torch.tensor(200.))

    func = NormalInit(
        mean=300, std=1e-5, bias_prob=0.01, layer=['Conv2d', 'Linear'])
    res = bias_init_with_prob(0.01)
    func(model)
    assert model[0].weight.allclose(torch.tensor(300.))
    assert model[2].weight.allclose(torch.tensor(300.))
    assert model[0].bias.allclose(torch.tensor(res))
    assert model[2].bias.allclose(torch.tensor(res))


def test_uniforminit():
    """"test UniformInit class."""
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Linear(1, 2))
    func = UniformInit(a=1, b=1, bias=2)
    func(model)
    assert torch.equal(model[0].weight, torch.full(model[0].weight.shape, 1.))
    assert torch.equal(model[2].weight, torch.full(model[2].weight.shape, 1.))
    assert torch.equal(model[0].bias, torch.full(model[0].bias.shape, 2.))
    assert torch.equal(model[2].bias, torch.full(model[2].bias.shape, 2.))

    func = UniformInit(a=100, b=100, layer=['Conv2d', 'Linear'], bias=10)
    func(model)
    assert torch.equal(model[0].weight, torch.full(model[0].weight.shape,
                                                   100.))
    assert torch.equal(model[2].weight, torch.full(model[2].weight.shape,
                                                   100.))
    assert torch.equal(model[0].bias, torch.full(model[0].bias.shape, 10.))
    assert torch.equal(model[2].bias, torch.full(model[2].bias.shape, 10.))


def test_kaiminginit():
    """test KaimingInit class."""
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Linear(1, 2))
    func = KaimingInit(bias=0.1, layer='Conv2d')
    func(model)
    assert torch.equal(model[0].bias, torch.full(model[0].bias.shape, 0.1))
    assert not torch.equal(model[2].bias, torch.full(model[2].bias.shape, 0.1))

    func = KaimingInit(a=100, bias=10)
    constant_func = ConstantInit(val=0, bias=0)
    model.apply(constant_func)
    assert torch.equal(model[0].weight, torch.full(model[0].weight.shape, 0.))
    assert torch.equal(model[2].weight, torch.full(model[2].weight.shape, 0.))
    assert torch.equal(model[0].bias, torch.full(model[0].bias.shape, 0.))
    assert torch.equal(model[2].bias, torch.full(model[2].bias.shape, 0.))

    func(model)
    assert not torch.equal(model[0].weight,
                           torch.full(model[0].weight.shape, 0.))
    assert not torch.equal(model[2].weight,
                           torch.full(model[2].weight.shape, 0.))
    assert torch.equal(model[0].bias, torch.full(model[0].bias.shape, 10.))
    assert torch.equal(model[2].bias, torch.full(model[2].bias.shape, 10.))


class FooModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)
        self.conv2d = nn.Conv2d(3, 1, 3)
        self.conv2d_2 = nn.Conv2d(3, 2, 3)


def test_pretrainedinit():
    """test PretrainedInit class."""

    modelA = FooModule()
    constant_func = ConstantInit(val=1, bias=2)
    modelA.apply(constant_func)
    modelB = FooModule()
    funcB = PretrainedInit(checkpoint='modelA.pth')
    modelC = nn.Linear(1, 2)
    funcC = PretrainedInit(checkpoint='modelA.pth', prefix='linear.')
    with TemporaryDirectory():
        torch.save(modelA.state_dict(), 'modelA.pth')
        funcB(modelB)
        assert torch.equal(modelB.linear.weight,
                           torch.full(modelB.linear.weight.shape, 1.))
        assert torch.equal(modelB.linear.bias,
                           torch.full(modelB.linear.bias.shape, 2.))
        assert torch.equal(modelB.conv2d.weight,
                           torch.full(modelB.conv2d.weight.shape, 1.))
        assert torch.equal(modelB.conv2d.bias,
                           torch.full(modelB.conv2d.bias.shape, 2.))
        assert torch.equal(modelB.conv2d_2.weight,
                           torch.full(modelB.conv2d_2.weight.shape, 1.))
        assert torch.equal(modelB.conv2d_2.bias,
                           torch.full(modelB.conv2d_2.bias.shape, 2.))

        funcC(modelC)
        assert torch.equal(modelC.weight, torch.full(modelC.weight.shape, 1.))
        assert torch.equal(modelC.bias, torch.full(modelC.bias.shape, 2.))


def test_initialize():
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Linear(1, 2))
    foonet = FooModule()

    init_cfg = dict(type='Constant', val=1, bias=2)
    initialize(model, init_cfg)
    assert torch.equal(model[0].weight, torch.full(model[0].weight.shape, 1.))
    assert torch.equal(model[2].weight, torch.full(model[2].weight.shape, 1.))
    assert torch.equal(model[0].bias, torch.full(model[0].bias.shape, 2.))
    assert torch.equal(model[2].bias, torch.full(model[2].bias.shape, 2.))

    init_cfg = [
        dict(type='Constant', layer='Conv1d', val=1, bias=2),
        dict(type='Constant', layer='Linear', val=3, bias=4)
    ]
    initialize(model, init_cfg)
    assert torch.equal(model[0].weight, torch.full(model[0].weight.shape, 1.))
    assert torch.equal(model[2].weight, torch.full(model[2].weight.shape, 3.))
    assert torch.equal(model[0].bias, torch.full(model[0].bias.shape, 2.))
    assert torch.equal(model[2].bias, torch.full(model[2].bias.shape, 4.))

    init_cfg = dict(
        type='Constant',
        val=1,
        bias=2,
        layer=['Conv2d', 'Linear'],
        override=dict(type='Constant', name='conv2d_2', val=3, bias=4))
    initialize(foonet, init_cfg)
    assert torch.equal(foonet.linear.weight,
                       torch.full(foonet.linear.weight.shape, 1.))
    assert torch.equal(foonet.linear.bias,
                       torch.full(foonet.linear.bias.shape, 2.))
    assert torch.equal(foonet.conv2d.weight,
                       torch.full(foonet.conv2d.weight.shape, 1.))
    assert torch.equal(foonet.conv2d.bias,
                       torch.full(foonet.conv2d.bias.shape, 2.))
    assert torch.equal(foonet.conv2d_2.weight,
                       torch.full(foonet.conv2d_2.weight.shape, 3.))
    assert torch.equal(foonet.conv2d_2.bias,
                       torch.full(foonet.conv2d_2.bias.shape, 4.))

    init_cfg = dict(
        type='Pretrained',
        checkpoint='modelA.pth',
        override=dict(type='Constant', name='conv2d_2', val=3, bias=4))
    modelA = FooModule()
    constant_func = ConstantInit(val=1, bias=2)
    modelA.apply(constant_func)
    with TemporaryDirectory():
        torch.save(modelA.state_dict(), 'modelA.pth')
        initialize(foonet, init_cfg)
        assert torch.equal(foonet.linear.weight,
                           torch.full(foonet.linear.weight.shape, 1.))
        assert torch.equal(foonet.linear.bias,
                           torch.full(foonet.linear.bias.shape, 2.))
        assert torch.equal(foonet.conv2d.weight,
                           torch.full(foonet.conv2d.weight.shape, 1.))
        assert torch.equal(foonet.conv2d.bias,
                           torch.full(foonet.conv2d.bias.shape, 2.))
        assert torch.equal(foonet.conv2d_2.weight,
                           torch.full(foonet.conv2d_2.weight.shape, 3.))
        assert torch.equal(foonet.conv2d_2.bias,
                           torch.full(foonet.conv2d_2.bias.shape, 4.))
    # test init_cfg type
    with pytest.raises(TypeError):
        init_cfg = 'init_cfg'
        initialize(foonet, init_cfg)

    # test override value type
    with pytest.raises(TypeError):
        init_cfg = dict(
            type='Constant',
            val=1,
            bias=2,
            layer=['Conv2d', 'Linear'],
            override='conv')
        initialize(foonet, init_cfg)

    # test override name
    with pytest.raises(RuntimeError):
        init_cfg = dict(
            type='Constant',
            val=1,
            bias=2,
            layer=['Conv2d', 'Linear'],
            override=dict(type='Constant', name='conv2d_3', val=3, bias=4))
        initialize(foonet, init_cfg)

    # test list override name
    with pytest.raises(RuntimeError):
        init_cfg = dict(
            type='Constant',
            val=1,
            bias=2,
            layer=['Conv2d', 'Linear'],
            override=[
                dict(type='Constant', name='conv2d', val=3, bias=4),
                dict(type='Constant', name='conv2d_3', val=5, bias=6)
            ])
        initialize(foonet, init_cfg)
