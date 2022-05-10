# Copyright (c) OpenMMLab. All rights reserved.
import random
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
from scipy import stats
from torch import nn

from mmcv.cnn import (Caffe2XavierInit, ConstantInit, KaimingInit, NormalInit,
                      PretrainedInit, TruncNormalInit, UniformInit, XavierInit,
                      bias_init_with_prob, caffe2_xavier_init, constant_init,
                      initialize, kaiming_init, normal_init, trunc_normal_init,
                      uniform_init, xavier_init)

if torch.__version__ == 'parrots':
    pytest.skip('not supported in parrots now', allow_module_level=True)


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


def test_trunc_normal_init():

    def _random_float(a, b):
        return (b - a) * random.random() + a

    def _is_trunc_normal(tensor, mean, std, a, b):
        # scipy's trunc norm is suited for data drawn from N(0, 1),
        # so we need to transform our data to test it using scipy.
        z_samples = (tensor.view(-1) - mean) / std
        z_samples = z_samples.tolist()
        a0 = (a - mean) / std
        b0 = (b - mean) / std
        p_value = stats.kstest(z_samples, 'truncnorm', args=(a0, b0))[1]
        return p_value > 0.0001

    conv_module = nn.Conv2d(3, 16, 3)
    mean = _random_float(-3, 3)
    std = _random_float(.01, 1)
    a = _random_float(mean - 2 * std, mean)
    b = _random_float(mean, mean + 2 * std)
    trunc_normal_init(conv_module, mean, std, a, b, bias=0.1)
    assert _is_trunc_normal(conv_module.weight, mean, std, a, b)
    assert conv_module.bias.allclose(torch.full_like(conv_module.bias, 0.1))

    conv_module_no_bias = nn.Conv2d(3, 16, 3, bias=False)
    trunc_normal_init(conv_module_no_bias)
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

    # test layer key with base class name
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Conv1d(1, 2, 1))
    func = ConstantInit(val=4., bias=5., layer='_ConvNd')
    func(model)
    assert torch.all(model[0].weight == 4.)
    assert torch.all(model[2].weight == 4.)
    assert torch.all(model[0].bias == 5.)
    assert torch.all(model[2].bias == 5.)

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

    constant_func = ConstantInit(val=0, bias=0, layer=['Conv2d', 'Linear'])
    func = XavierInit(gain=100, bias_prob=0.01, layer=['Conv2d', 'Linear'])
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

    # test layer key with base class name
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Conv1d(1, 2, 1))
    func = ConstantInit(val=4., bias=5., layer='_ConvNd')
    func(model)
    assert torch.all(model[0].weight == 4.)
    assert torch.all(model[2].weight == 4.)
    assert torch.all(model[0].bias == 5.)
    assert torch.all(model[2].bias == 5.)

    func = XavierInit(gain=100, bias_prob=0.01, layer='_ConvNd')
    func(model)
    assert not torch.all(model[0].weight == 4.)
    assert not torch.all(model[2].weight == 4.)
    assert torch.all(model[0].bias == res)
    assert torch.all(model[2].bias == res)

    # test bias input type
    with pytest.raises(TypeError):
        func = XavierInit(bias='0.1', layer='Conv2d')
    # test layer inpur type
    with pytest.raises(TypeError):
        func = XavierInit(bias=0.1, layer=1)


def test_normalinit():
    """test Normalinit class."""
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Linear(1, 2))

    func = NormalInit(mean=100, std=1e-5, bias=200, layer=['Conv2d', 'Linear'])
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

    # test layer key with base class name
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Conv1d(1, 2, 1))

    func = NormalInit(mean=300, std=1e-5, bias_prob=0.01, layer='_ConvNd')
    func(model)
    assert model[0].weight.allclose(torch.tensor(300.))
    assert model[2].weight.allclose(torch.tensor(300.))
    assert torch.all(model[0].bias == res)
    assert torch.all(model[2].bias == res)


def test_truncnormalinit():
    """test TruncNormalInit class."""
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Linear(1, 2))

    func = TruncNormalInit(
        mean=100, std=1e-5, bias=200, a=0, b=200, layer=['Conv2d', 'Linear'])
    func(model)
    assert model[0].weight.allclose(torch.tensor(100.))
    assert model[2].weight.allclose(torch.tensor(100.))
    assert model[0].bias.allclose(torch.tensor(200.))
    assert model[2].bias.allclose(torch.tensor(200.))

    func = TruncNormalInit(
        mean=300,
        std=1e-5,
        a=100,
        b=400,
        bias_prob=0.01,
        layer=['Conv2d', 'Linear'])
    res = bias_init_with_prob(0.01)
    func(model)
    assert model[0].weight.allclose(torch.tensor(300.))
    assert model[2].weight.allclose(torch.tensor(300.))
    assert model[0].bias.allclose(torch.tensor(res))
    assert model[2].bias.allclose(torch.tensor(res))

    # test layer key with base class name
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Conv1d(1, 2, 1))

    func = TruncNormalInit(
        mean=300, std=1e-5, a=100, b=400, bias_prob=0.01, layer='_ConvNd')
    func(model)
    assert model[0].weight.allclose(torch.tensor(300.))
    assert model[2].weight.allclose(torch.tensor(300.))
    assert torch.all(model[0].bias == res)
    assert torch.all(model[2].bias == res)


def test_uniforminit():
    """"test UniformInit class."""
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Linear(1, 2))
    func = UniformInit(a=1, b=1, bias=2, layer=['Conv2d', 'Linear'])
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

    # test layer key with base class name
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Conv1d(1, 2, 1))

    func = UniformInit(a=100, b=100, bias_prob=0.01, layer='_ConvNd')
    res = bias_init_with_prob(0.01)
    func(model)
    assert torch.all(model[0].weight == 100.)
    assert torch.all(model[2].weight == 100.)
    assert torch.all(model[0].bias == res)
    assert torch.all(model[2].bias == res)


def test_kaiminginit():
    """test KaimingInit class."""
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Linear(1, 2))
    func = KaimingInit(bias=0.1, layer='Conv2d')
    func(model)
    assert torch.equal(model[0].bias, torch.full(model[0].bias.shape, 0.1))
    assert not torch.equal(model[2].bias, torch.full(model[2].bias.shape, 0.1))

    func = KaimingInit(a=100, bias=10, layer=['Conv2d', 'Linear'])
    constant_func = ConstantInit(val=0, bias=0, layer=['Conv2d', 'Linear'])
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

    # test layer key with base class name
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Conv1d(1, 2, 1))
    func = KaimingInit(bias=0.1, layer='_ConvNd')
    func(model)
    assert torch.all(model[0].bias == 0.1)
    assert torch.all(model[2].bias == 0.1)

    func = KaimingInit(a=100, bias=10, layer='_ConvNd')
    constant_func = ConstantInit(val=0, bias=0, layer='_ConvNd')
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


def test_caffe2xavierinit():
    """test Caffe2XavierInit."""
    model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU(), nn.Linear(1, 2))
    func = Caffe2XavierInit(bias=0.1, layer='Conv2d')
    func(model)
    assert torch.equal(model[0].bias, torch.full(model[0].bias.shape, 0.1))
    assert not torch.equal(model[2].bias, torch.full(model[2].bias.shape, 0.1))


class FooModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)
        self.conv2d = nn.Conv2d(3, 1, 3)
        self.conv2d_2 = nn.Conv2d(3, 2, 3)


def test_pretrainedinit():
    """test PretrainedInit class."""

    modelA = FooModule()
    constant_func = ConstantInit(val=1, bias=2, layer=['Conv2d', 'Linear'])
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

    # test layer key
    init_cfg = dict(type='Constant', layer=['Conv2d', 'Linear'], val=1, bias=2)
    initialize(model, init_cfg)
    assert torch.equal(model[0].weight, torch.full(model[0].weight.shape, 1.))
    assert torch.equal(model[2].weight, torch.full(model[2].weight.shape, 1.))
    assert torch.equal(model[0].bias, torch.full(model[0].bias.shape, 2.))
    assert torch.equal(model[2].bias, torch.full(model[2].bias.shape, 2.))
    assert init_cfg == dict(
        type='Constant', layer=['Conv2d', 'Linear'], val=1, bias=2)

    # test init_cfg with list type
    init_cfg = [
        dict(type='Constant', layer='Conv2d', val=1, bias=2),
        dict(type='Constant', layer='Linear', val=3, bias=4)
    ]
    initialize(model, init_cfg)
    assert torch.equal(model[0].weight, torch.full(model[0].weight.shape, 1.))
    assert torch.equal(model[2].weight, torch.full(model[2].weight.shape, 3.))
    assert torch.equal(model[0].bias, torch.full(model[0].bias.shape, 2.))
    assert torch.equal(model[2].bias, torch.full(model[2].bias.shape, 4.))
    assert init_cfg == [
        dict(type='Constant', layer='Conv2d', val=1, bias=2),
        dict(type='Constant', layer='Linear', val=3, bias=4)
    ]

    # test layer key and override key
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
    assert init_cfg == dict(
        type='Constant',
        val=1,
        bias=2,
        layer=['Conv2d', 'Linear'],
        override=dict(type='Constant', name='conv2d_2', val=3, bias=4))

    # test override key
    init_cfg = dict(
        type='Constant', val=5, bias=6, override=dict(name='conv2d_2'))
    initialize(foonet, init_cfg)
    assert not torch.equal(foonet.linear.weight,
                           torch.full(foonet.linear.weight.shape, 5.))
    assert not torch.equal(foonet.linear.bias,
                           torch.full(foonet.linear.bias.shape, 6.))
    assert not torch.equal(foonet.conv2d.weight,
                           torch.full(foonet.conv2d.weight.shape, 5.))
    assert not torch.equal(foonet.conv2d.bias,
                           torch.full(foonet.conv2d.bias.shape, 6.))
    assert torch.equal(foonet.conv2d_2.weight,
                       torch.full(foonet.conv2d_2.weight.shape, 5.))
    assert torch.equal(foonet.conv2d_2.bias,
                       torch.full(foonet.conv2d_2.bias.shape, 6.))
    assert init_cfg == dict(
        type='Constant', val=5, bias=6, override=dict(name='conv2d_2'))

    init_cfg = dict(
        type='Pretrained',
        checkpoint='modelA.pth',
        override=dict(type='Constant', name='conv2d_2', val=3, bias=4))
    modelA = FooModule()
    constant_func = ConstantInit(val=1, bias=2, layer=['Conv2d', 'Linear'])
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
    assert init_cfg == dict(
        type='Pretrained',
        checkpoint='modelA.pth',
        override=dict(type='Constant', name='conv2d_2', val=3, bias=4))

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

    # test override with args except type key
    with pytest.raises(ValueError):
        init_cfg = dict(
            type='Constant',
            val=1,
            bias=2,
            override=dict(name='conv2d_2', val=3, bias=4))
        initialize(foonet, init_cfg)

    # test override without name
    with pytest.raises(ValueError):
        init_cfg = dict(
            type='Constant',
            val=1,
            bias=2,
            override=dict(type='Constant', val=3, bias=4))
        initialize(foonet, init_cfg)
