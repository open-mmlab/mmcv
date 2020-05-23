# Copyright (c) Open-MMLab. All rights reserved.
import numpy as np
import pytest
import torch
from torch import nn

from mmcv.cnn import (bias_init_with_prob, caffe2_xavier_init, constant_init,
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
