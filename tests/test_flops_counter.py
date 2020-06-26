import pytest
import torch
import torch.nn as nn

from mmcv.cnn import get_model_complexity_info
from mmcv.cnn.utils.flops_counter import (add_flops_mask,
                                          conv_flops_counter_hook,
                                          flops_to_string, params_to_string)

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

# yapf: disable
gt_results = [{'model': 'nn.Linear(1024, 2)', 'input': (1, 1024), 'flops': 1.0, 'params': 2050.0},      # noqa: E501
              {'model': 'nn.Conv1d(3, 8, 3)', 'input': (3, 16), 'flops': 1120.0, 'params': 80.0},       # noqa: E501
              {'model': 'nn.Conv2d(3, 8, 3)', 'input': (3, 16, 16), 'flops': 43904.0, 'params': 224.0},     # noqa: E501
              {'model': 'nn.Conv3d(3, 8, 3)', 'input': (3, 3, 16, 16), 'flops': 128576.0, 'params': 656.0},     # noqa: E501
              {'model': 'nn.ConvTranspose1d(3, 8, 3)', 'input': (3, 16), 'flops': 1440.0, 'params': 80.0},  # noqa: E501
              {'model': 'nn.ConvTranspose2d(3, 8, 3)', 'input': (3, 16, 16), 'flops': 72576.0, 'params': 224.0},    # noqa: E501
              {'model': 'nn.ConvTranspose3d(3, 8, 3)', 'input': (3, 3, 16, 16), 'flops': 1062720.0, 'params': 656.0},   # noqa: E501
              {'model': 'nn.BatchNorm1d(3, 8)', 'input': (3, 16), 'flops': 96.0, 'params': 6.0},    # noqa: E501
              {'model': 'nn.BatchNorm2d(3, 8)', 'input': (3, 16, 16), 'flops': 1536.0, 'params': 6.0},  # noqa: E501
              {'model': 'nn.BatchNorm3d(3, 8)', 'input': (3, 3, 16, 16), 'flops': 4608.0, 'params': 6.0},   # noqa: E501
              {'model': 'nn.SyncBatchNorm(8)', 'input': (8, 16, 16), 'flops': 4096.0, 'params': 16.0},  # noqa: E501
              {'model': 'nn.GroupNorm(2, 4)', 'input': (4, 16, 16), 'flops': 4096.0, 'params': 8},      # noqa: E501
              {'model': 'nn.Sequential(nn.Conv2d(3, 8, 3), nn.Upsample((32, 32)))', 'input': (3, 16, 16), 'flops': 52096.0, 'params': 224},     # noqa: E501
              {'model': 'nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU())', 'input': (3, 16, 16), 'flops': 45472.0, 'params': 224},     # noqa: E501
              {'model': 'nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU6())', 'input': (3, 16, 16), 'flops': 45472.0, 'params': 224},    # noqa: E501
              {'model': 'nn.Sequential(nn.Conv2d(3, 8, 3), nn.LeakyReLU())', 'input': (3, 16, 16), 'flops': 45472.0, 'params': 224},    # noqa: E501
              {'model': 'nn.Sequential(nn.Conv2d(3, 8, 3), nn.ELU())', 'input': (3, 16, 16), 'flops': 45472.0, 'params': 224},  # noqa: E501
              {'model': 'nn.Sequential(nn.Conv2d(3, 8, 3), nn.PReLU())', 'input': (3, 16, 16), 'flops': 45472.0, 'params': 225},    # noqa: E501
              {'model': 'nn.Sequential(nn.Conv2d(3, 8, 3), nn.AdaptiveMaxPool2d(8))', 'input': (3, 16, 16), 'flops': 45472.0, 'params': 224},   # noqa: E501
              {'model': 'nn.Sequential(nn.Conv2d(3, 8, 3), nn.AdaptiveAvgPool2d(8))', 'input': (3, 16, 16), 'flops': 45472.0, 'params': 224},   # noqa: E501
              {'model': 'nn.Sequential(nn.Conv2d(3, 8, 3), nn.AvgPool2d(2))', 'input': (3, 16, 16), 'flops': 45472.0, 'params': 224},   # noqa: E501
              {'model': 'nn.Sequential(nn.Conv2d(3, 8, 3), nn.MaxPool2d(2))', 'input': (3, 16, 16), 'flops': 45472.0, 'params': 224},   # noqa: E501
              {'model': 'nn.Sequential(nn.Conv2d(3, 8, 3), nn.Upsample((32, 32)))', 'input': (3, 16, 16), 'flops': 52096.0, 'params': 224}]    # noqa: E501
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
        input_res = (256, )
        get_model_complexity_info(model, input_res)

    # test common layers
    for item in gt_results:
        model = item['model']
        input = item['input']
        if 'SyncBatchNorm' in model:
            if torch.cuda.is_available():
                model = eval(model).cuda()
            else:
                continue
        else:
            model = eval(model)
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
    assert flops == '0.17 GMac' and params == str(656)

    # test print per layer status
    model = nn.Conv1d(3, 8, 3)
    x = (3, 16)
    out = StringIO()
    get_model_complexity_info(model, x, ost=out)
    assert out.getvalue(
    ) == 'Conv1d(0.0 GMac, 100.000% MACs, 3, 8, kernel_size=(3,), stride=(1,))\n'  # noqa: E501

    # test when model is not a common instance
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.Flatten(), nn.Linear(1568, 2))
    x = (3, 16, 16)
    flops, params = get_model_complexity_info(
        model, x, as_strings=False, print_per_layer_stat=False)
    assert flops == 47040.0 and params == 3362


def test_flops_to_string():
    flops = 6.54321 * 10.**9
    assert flops_to_string(flops) == '6.54 GMac'
    assert flops_to_string(flops, 'MMac') == '6543.21 MMac'
    assert flops_to_string(flops, 'KMac') == '6543210.0 KMac'
    assert flops_to_string(flops, 'Mac') == '6543210000.0 Mac'
    assert flops_to_string(flops, precision=4) == '6.5432 GMac'

    flops = 6.54321 * 10.**9
    assert flops_to_string(flops, None) == '6.54 GMac'
    flops = 3.21 * 10.**7
    assert flops_to_string(flops, None) == '32.1 MMac'
    flops = 5.4 * 10.**3
    assert flops_to_string(flops, None) == '5.4 KMac'
    flops = 987
    assert flops_to_string(flops, None) == '987 Mac'


def test_params_to_string():
    params_num = 3.21 * 10.**7
    assert params_to_string(params_num) == '32.1 M'

    params_num = 4.56 * 10.**5
    assert params_to_string(params_num) == '456.0 k'

    params_num = 7.89 * 10.**2
    assert params_to_string(params_num) == '789.0'


def test_add_mask():
    model = nn.Conv2d(3, 8, 3)
    mask = torch.ones(1, 1, 1, 1)
    add_flops_mask(model, mask)
    assert torch.equal(mask, model.__mask__)

    model.__flops__ = 0
    tensor_shape = (1, 3, 16, 16)
    input_tensor = torch.randn(tensor_shape)
    output_tensor = model(input_tensor)
    conv_flops_counter_hook(model, input_tensor, output_tensor)
    assert model.__flops__ == 131712
