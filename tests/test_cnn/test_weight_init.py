# Copyright (c) Open-MMLab. All rights reserved.
import numpy as np
import pytest
import torch
from torch import nn

from mmcv.cnn import (bias_init_with_prob, caffe2_xavier_init, constant_init,
                      initialize, kaiming_init, normal_init, pretrained_init,
                      random_init, uniform_init, xavier_init)


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


def test_initialize():
    model = nn.Sequential(nn.Conv2d(2, 5, 3), nn.ReLU(), nn.Linear(20, 10))
    init_cfg = dict(
        type='random',
        Conv2d=[
            dict(function='constant', parameters='weight', val=1),
            dict(function='constant', parameters='bias', val=2)
        ],
        Linear=dict(function='constant_init', val=3, bias=4))

    initialize(model, init_cfg)
    for i, parameter in enumerate(model.parameters()):
        assert parameter.allclose(torch.tensor(float(i + 1)))
    init_cfg = 'torchvision://resnet50'
    with pytest.raises(TypeError):
        initialize(model, init_cfg)
    init_cfg = dict()
    with pytest.raises(TypeError):
        initialize(model, init_cfg)
    init_cfg = init_cfg = dict(
        type='init', Linear=dict(function='constant_init', val=3, bias=4))
    with pytest.raises(TypeError):
        initialize(model, init_cfg)


def test_random_init():
    model = nn.Conv2d(2, 5, 3, bias=True)
    init_cfg = dict(
        type='random',
        Conv2d=[
            dict(function='constant', parameters='weight', val=1),
            dict(function='constant', parameters='bias', val=2)
        ])
    random_init(model, init_cfg)
    assert model.weight.allclose(torch.tensor(1.0))
    assert model.bias.allclose(torch.tensor(2.0))
    init_cfg = dict(
        type='random', Conv2d=dict(function='constant_init', val=3, bias=4))
    random_init(model, init_cfg)
    assert model.weight.allclose(torch.tensor(3.0))
    assert model.bias.allclose(torch.tensor(4.0))


def test_pretrained_init():
    from mmdet.models import ResNet
    model = ResNet(
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
    )
    url = 'torchvision://resnet50'
    init_cfg = dict(type='pretrained', checkpoint=url)
    pretrained_init(model, init_cfg)
    from torchvision.models import resnet50
    model_from_torchvision = resnet50(pretrained=True)
    for (name,
         p), (name_torchvision,
              p_torchvision) in zip(model.named_parameters(),
                                    model_from_torchvision.named_parameters()):
        assert name == name_torchvision
        assert torch.equal(p, p_torchvision)

    url = 'http://download.openmmlab.com/mmdetection/v2.0/retinanet/'\
        'retinanet_r50_fpn_1x_coco/'\
        'retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
    init_cfg = dict(type='pretrained', checkpoint=url, prefix='backbone.')
    pretrained_init(model, init_cfg)

    init_cfg = dict(type='pretrained')
    with pytest.raises(TypeError):
        pretrained_init(model, init_cfg)
