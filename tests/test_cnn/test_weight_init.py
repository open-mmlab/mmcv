# Copyright (c) Open-MMLab. All rights reserved.
from sys import prefix
import numpy as np
from tempfile import TemporaryDirectory
import pytest
import torch
from torch import nn
from torch.nn.modules.conv import Conv1d
from torch.serialization import check_module_version_greater_or_equal

from mmcv.cnn import (bias_init_with_prob, caffe2_xavier_init, constant_init,
                      initialize, kaiming_init, normal_init, uniform_init,
                      xavier_init, ConstantInit, XavierInit, NormalInit,
                      UniformInit, KaimingInit, BiasInitWithProb,
                      PretrainedInit)


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
    """test ConstantInit class"""
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Linear(1, 2))
    func = ConstantInit(val=1, bias=2, layers='Conv2d')
    model.apply(func)
    assert model[0].weight.allclose(torch.tensor(1.0), rtol=0)
    assert model[0].bias.allclose(torch.tensor(2.0), rtol=0)
    assert not model[2].weight.allclose(torch.tensor(1.0), rtol=0)
    assert not model[2].bias.allclose(torch.tensor(2.0), rtol=0)

    func = ConstantInit(
        val=3,
        bias=dict(type='BiasInitWithProb', prior_prob=0.01),
        layers='Linear')
    model.apply(func)
    res = bias_init_with_prob(0.01)
    assert model[0].weight.allclose(torch.tensor(1.0))
    assert model[2].weight.allclose(torch.tensor(3.0))
    assert model[0].bias.allclose(torch.tensor(2.0))
    assert model[2].bias.allclose(torch.tensor(res))

    func = ConstantInit(val=4, bias=5)
    model.apply(func)
    assert model[0].weight.allclose(torch.tensor(4.0))
    assert model[2].weight.allclose(torch.tensor(4.0))
    assert model[0].bias.allclose(torch.tensor(5.0))
    assert model[2].bias.allclose(torch.tensor(5.0))

    with pytest.raises(TypeError):
        func = ConstantInit(val=1, bias='1')


def test_xavierinit():
    """test XavierInit class """
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Linear(1, 2))
    func = XavierInit(bias=0.1, layers='Conv2d')
    model.apply(func)
    assert model[0].bias.allclose(torch.full_like(model[2].bias, 0.1))
    assert not model[2].bias.allclose(torch.full_like(model[0].bias, 0.1))

    constant_func = ConstantInit(val=0, bias=0)
    func = XavierInit(
        gain=100, bias=dict(type='BiasInitWithProb', prior_prob=0.01))
    model.apply(constant_func)
    assert model[0].weight.allclose(torch.tensor(0.0))
    assert model[2].weight.allclose(torch.tensor(0.0))
    assert model[0].bias.allclose(torch.tensor(0.0))
    assert model[2].bias.allclose(torch.tensor(0.0))
    res = bias_init_with_prob(0.01)
    model.apply(func)
    assert not model[0].weight.allclose(torch.tensor(0.0))
    assert not model[2].weight.allclose(torch.tensor(0.0))
    assert model[0].bias.allclose(torch.tensor(res))
    assert model[2].bias.allclose(torch.tensor(res))


def test_normalinit():
    """test Normalinit class"""
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Linear(1, 2))

    func = NormalInit(mean=100, std=1e-5, bias=200)
    model.apply(func)
    assert model[0].weight.allclose(torch.tensor(100.0))
    assert model[2].weight.allclose(torch.tensor(100.0))
    assert model[0].bias.allclose(torch.tensor(200.0), rtol=0)
    assert model[2].bias.allclose(torch.tensor(200.0), rtol=0)

    func = NormalInit(
        mean=300,
        std=1e-5,
        bias=dict(type='BiasInitWithProb', prior_prob=0.01),
        layers=['Conv2d', 'Linear'])
    res = bias_init_with_prob(0.01)
    model.apply(func)
    assert model[0].weight.allclose(torch.tensor(300.0))
    assert model[2].weight.allclose(torch.tensor(300.0))
    assert model[0].bias.allclose(torch.tensor(res), rtol=0)
    assert model[2].bias.allclose(torch.tensor(res), rtol=0)


def test_uniforminit():
    """"test UniformInit class"""
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Linear(1, 2))
    func = UniformInit(a=1, b=1, bias=2)
    model.apply(func)
    assert model[0].weight.allclose(torch.tensor(1.0))
    assert model[2].weight.allclose(torch.tensor(1.0))
    assert model[0].bias.allclose(torch.tensor(2.0))
    assert model[2].bias.allclose(torch.tensor(2.0))

    func = UniformInit(a=100, b=100, layers=['Conv2d', 'Linear'], bias=10)
    model.apply(func)
    assert model[0].weight.allclose(torch.tensor(100.0))
    assert model[2].weight.allclose(torch.tensor(100.0))
    assert model[0].bias.allclose(torch.tensor(10.0))
    assert model[2].bias.allclose(torch.tensor(10.0))


def test_kaiminginit():
    """test KaimingInit class"""
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Linear(1, 2))
    func = KaimingInit(bias=0.1, layers='Conv2d')
    model.apply(func)
    assert model[0].bias.allclose(torch.full_like(model[2].bias, 0.1))
    assert not model[2].bias.allclose(torch.full_like(model[0].bias, 0.1))

    func = KaimingInit(a=100, bias=10)
    constant_func = ConstantInit(val=0, bias=0)
    model.apply(constant_func)
    assert model[0].weight.allclose(torch.tensor(0.0))
    assert model[2].weight.allclose(torch.tensor(0.0))
    assert model[0].bias.allclose(torch.tensor(0.0))
    assert model[2].bias.allclose(torch.tensor(0.0))

    model.apply(func)
    assert not model[0].weight.allclose(torch.tensor(0.0))
    assert not model[2].weight.allclose(torch.tensor(0.0))
    assert model[0].bias.allclose(torch.tensor(10.0))
    assert model[2].bias.allclose(torch.tensor(10.0))


def test_biasinitwithprob():
    """test BiasInitWithProb class"""
    func = BiasInitWithProb(0.5)
    res = func()
    assert res == bias_init_with_prob(0.5)


class FooModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)
        self.conv2d = nn.Conv2d(3, 1, 3)
        self.conv2d_2 = nn.Conv2d(3, 2, 3)


def test_pretrainedinit():
    """test PretrainedInit class"""

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
        assert modelB.linear.weight.allclose(torch.tensor(1.0))
        assert modelB.linear.bias.allclose(torch.tensor(2.0))
        assert modelB.conv2d.weight.allclose(torch.tensor(1.0))
        assert modelB.conv2d.bias.allclose(torch.tensor(2.0))
        assert modelB.conv2d_2.weight.allclose(torch.tensor(1.0))
        assert modelB.conv2d_2.bias.allclose(torch.tensor(2.0))
        funcC(modelC)
        assert modelC.weight.allclose(torch.tensor(1.0))
        assert modelC.bias.allclose(torch.tensor(2.0))


def test_initialize():
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Linear(1, 2))
    foonet = FooModule()

    init_cfg = dict(type='ConstantInit', val=1, bias=2)
    initialize(model, init_cfg)
    assert model[0].weight.allclose(torch.tensor(1.0))
    assert model[2].weight.allclose(torch.tensor(1.0))
    assert model[0].bias.allclose(torch.tensor(2.0))
    assert model[2].bias.allclose(torch.tensor(2.0))

    init_cfg = [
        dict(type='ConstantInit', layers='Conv1d', val=1, bias=2),
        dict(type='ConstantInit', layers='Linear', val=3, bias=4)
    ]
    initialize(model, init_cfg)
    assert model[0].weight.allclose(torch.tensor(1.0))
    assert model[2].weight.allclose(torch.tensor(3.0))
    assert model[0].bias.allclose(torch.tensor(2.0))
    assert model[2].bias.allclose(torch.tensor(4.0))

    init_cfg = dict(
        type='ConstantInit',
        val=1,
        bias=2,
        layers=['Conv2d', 'Linear'],
        cases=dict(type='ConstantInit', name='conv2d_2', val=3, bias=4))
    initialize(foonet, init_cfg)
    assert foonet.linear.weight.allclose(torch.tensor(1.0))
    assert foonet.linear.bias.allclose(torch.tensor(2.0))
    assert foonet.conv2d.weight.allclose(torch.tensor(1.0))
    assert foonet.conv2d.bias.allclose(torch.tensor(2.0))
    assert foonet.conv2d_2.weight.allclose(torch.tensor(3.0))
    assert foonet.conv2d_2.bias.allclose(torch.tensor(4.0))

    init_cfg = dict(
        type='PretrainedInit',
        checkpoint='modelA.pth',
        cases=dict(type='ConstantInit', name='conv2d_2', val=3, bias=4))
    modelA = FooModule()
    constant_func = ConstantInit(val=1, bias=2)
    modelA.apply(constant_func)
    with TemporaryDirectory():
        torch.save(modelA.state_dict(), 'modelA.pth')
        initialize(foonet, init_cfg)
        assert foonet.linear.weight.allclose(torch.tensor(1.0))
        assert foonet.linear.bias.allclose(torch.tensor(2.0))
        assert foonet.conv2d.weight.allclose(torch.tensor(1.0))
        assert foonet.conv2d.bias.allclose(torch.tensor(2.0))
        assert foonet.conv2d_2.weight.allclose(torch.tensor(3.0))
        assert foonet.conv2d_2.bias.allclose(torch.tensor(4.0))
