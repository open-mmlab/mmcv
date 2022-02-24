# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn

from mmcv.cnn import get_model_complexity_info
from mmcv.cnn.utils.flops_counter import flops_to_string, params_to_string

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

# yapf: disable
gt_results = [
    {'model': nn.Conv1d(3, 8, 3), 'input': (3, 16), 'flops': 1120.0, 'params': 80.0},  # noqa: E501
    {'model': nn.Conv2d(3, 8, 3), 'input': (3, 16, 16), 'flops': 43904.0, 'params': 224.0},  # noqa: E501
    {'model': nn.Conv3d(3, 8, 3), 'input': (3, 3, 16, 16), 'flops': 128576.0, 'params': 656.0},  # noqa: E501
    {'model': nn.ReLU(), 'input': (3, 16, 16), 'flops': 768.0, 'params': 0},  # noqa: E501
    {'model': nn.PReLU(), 'input': (3, 16, 16), 'flops': 768.0, 'params': 1},  # noqa: E501
    {'model': nn.ELU(), 'input': (3, 16, 16), 'flops': 768.0, 'params': 0},  # noqa: E501
    {'model': nn.LeakyReLU(), 'input': (3, 16, 16), 'flops': 768.0, 'params': 0},  # noqa: E501
    {'model': nn.ReLU6(), 'input': (3, 16, 16), 'flops': 768.0, 'params': 0},  # noqa: E501
    {'model': nn.MaxPool1d(2), 'input': (3, 16), 'flops': 48.0, 'params': 0},  # noqa: E501
    {'model': nn.MaxPool2d(2), 'input': (3, 16, 16), 'flops': 768.0, 'params': 0},  # noqa: E501
    {'model': nn.MaxPool3d(2), 'input': (3, 3, 16, 16), 'flops': 2304.0, 'params': 0},  # noqa: E501
    {'model': nn.AvgPool1d(2), 'input': (3, 16), 'flops': 48.0, 'params': 0},  # noqa: E501
    {'model': nn.AvgPool2d(2), 'input': (3, 16, 16), 'flops': 768.0, 'params': 0},  # noqa: E501
    {'model': nn.AvgPool3d(2), 'input': (3, 3, 16, 16), 'flops': 2304.0, 'params': 0},  # noqa: E501
    {'model': nn.AdaptiveMaxPool1d(2), 'input': (3, 16), 'flops': 48.0, 'params': 0},  # noqa: E501
    {'model': nn.AdaptiveMaxPool2d(2), 'input': (3, 16, 16), 'flops': 768.0, 'params': 0},  # noqa: E501
    {'model': nn.AdaptiveMaxPool3d(2), 'input': (3, 3, 16, 16), 'flops': 2304.0, 'params': 0},  # noqa: E501
    {'model': nn.AdaptiveAvgPool1d(2), 'input': (3, 16), 'flops': 48.0, 'params': 0},  # noqa: E501
    {'model': nn.AdaptiveAvgPool2d(2), 'input': (3, 16, 16), 'flops': 768.0, 'params': 0},  # noqa: E501
    {'model': nn.AdaptiveAvgPool3d(2), 'input': (3, 3, 16, 16), 'flops': 2304.0, 'params': 0},  # noqa: E501
    {'model': nn.BatchNorm1d(3), 'input': (3, 16), 'flops': 96.0, 'params': 6.0},  # noqa: E501
    {'model': nn.BatchNorm2d(3), 'input': (3, 16, 16), 'flops': 1536.0, 'params': 6.0},  # noqa: E501
    {'model': nn.BatchNorm3d(3), 'input': (3, 3, 16, 16), 'flops': 4608.0, 'params': 6.0},  # noqa: E501
    {'model': nn.GroupNorm(2, 6), 'input': (6, 16, 16), 'flops': 3072.0, 'params': 12.0},  # noqa: E501
    {'model': nn.InstanceNorm1d(3, affine=True), 'input': (3, 16), 'flops': 96.0, 'params': 6.0},  # noqa: E501
    {'model': nn.InstanceNorm2d(3, affine=True), 'input': (3, 16, 16), 'flops': 1536.0, 'params': 6.0},  # noqa: E501
    {'model': nn.InstanceNorm3d(3, affine=True), 'input': (3, 3, 16, 16), 'flops': 4608.0, 'params': 6.0},  # noqa: E501
    {'model': nn.LayerNorm((3, 16, 16)), 'input': (3, 16, 16), 'flops': 1536.0, 'params': 1536.0},  # noqa: E501
    {'model': nn.LayerNorm((3, 16, 16), elementwise_affine=False), 'input': (3, 16, 16), 'flops': 768.0, 'params': 0},  # noqa: E501
    {'model': nn.Linear(1024, 2), 'input': (1024, ), 'flops': 2048.0, 'params': 2050.0},  # noqa: E501
    {'model': nn.ConvTranspose2d(3, 8, 3), 'input': (3, 16, 16), 'flops': 57888, 'params': 224.0},  # noqa: E501
    {'model': nn.Upsample((32, 32)), 'input': (3, 16, 16), 'flops': 3072.0, 'params': 0}  # noqa: E501
]
# yapf: enable


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 8, 3)

    def forward(self, imgs):
        x = torch.randn((1, *imgs))
        return self.conv2d(x)


def input_constructor(x):
    return dict(imgs=x)


def test_flops_counter():
    with pytest.raises(AssertionError):
        # input_res should be a tuple
        model = nn.Conv2d(3, 8, 3)
        input_res = [1, 3, 16, 16]
        get_model_complexity_info(model, input_res)

    with pytest.raises(AssertionError):
        # len(input_res) >= 2
        model = nn.Conv2d(3, 8, 3)
        input_res = tuple()
        get_model_complexity_info(model, input_res)

    # test common layers
    for item in gt_results:
        model = item['model']
        input = item['input']
        flops, params = get_model_complexity_info(
            model, input, as_strings=False, print_per_layer_stat=False)
        assert flops == item['flops'] and params == item['params']

    # test input constructor
    model = ExampleModel()
    x = (3, 16, 16)
    flops, params = get_model_complexity_info(
        model,
        x,
        as_strings=False,
        print_per_layer_stat=False,
        input_constructor=input_constructor)
    assert flops == 43904.0 and params == 224.0

    # test output string
    model = nn.Conv3d(3, 8, 3)
    x = (3, 3, 512, 512)
    flops, params = get_model_complexity_info(
        model, x, print_per_layer_stat=False)
    assert flops == '0.17 GFLOPs' and params == str(656)

    # test print per layer status
    model = nn.Conv1d(3, 8, 3)
    x = (3, 16)
    out = StringIO()
    get_model_complexity_info(model, x, ost=out)
    assert out.getvalue() == \
        'Conv1d(0.0 M, 100.000% Params, 0.0 GFLOPs, 100.000% FLOPs, 3, 8, kernel_size=(3,), stride=(1,))\n'  # noqa: E501

    # test when model is not a common instance
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.Flatten(), nn.Linear(1568, 2))
    x = (3, 16, 16)
    flops, params = get_model_complexity_info(
        model, x, as_strings=False, print_per_layer_stat=True)
    assert flops == 47040.0 and params == 3362


def test_flops_to_string():
    flops = 6.54321 * 10.**9
    assert flops_to_string(flops) == '6.54 GFLOPs'
    assert flops_to_string(flops, 'MFLOPs') == '6543.21 MFLOPs'
    assert flops_to_string(flops, 'KFLOPs') == '6543210.0 KFLOPs'
    assert flops_to_string(flops, 'FLOPs') == '6543210000.0 FLOPs'
    assert flops_to_string(flops, precision=4) == '6.5432 GFLOPs'

    flops = 6.54321 * 10.**9
    assert flops_to_string(flops, None) == '6.54 GFLOPs'
    flops = 3.21 * 10.**7
    assert flops_to_string(flops, None) == '32.1 MFLOPs'
    flops = 5.4 * 10.**3
    assert flops_to_string(flops, None) == '5.4 KFLOPs'
    flops = 987
    assert flops_to_string(flops, None) == '987 FLOPs'


def test_params_to_string():
    num_params = 3.21 * 10.**7
    assert params_to_string(num_params) == '32.1 M'
    num_params = 4.56 * 10.**5
    assert params_to_string(num_params) == '456.0 k'
    num_params = 7.89 * 10.**2
    assert params_to_string(num_params) == '789.0'

    num_params = 6.54321 * 10.**7
    assert params_to_string(num_params, 'M') == '65.43 M'
    assert params_to_string(num_params, 'K') == '65432.1 K'
    assert params_to_string(num_params, '') == '65432100.0'
    assert params_to_string(num_params, precision=4) == '65.4321 M'
