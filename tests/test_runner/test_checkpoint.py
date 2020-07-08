from collections import OrderedDict

import torch.nn as nn
from torch.nn.parallel import DataParallel

from mmcv.parallel.registry import MODULE_WRAPPERS
from mmcv.runner.checkpoint import get_state_dict


@MODULE_WRAPPERS.register_module()
class DDPWrapper(object):

    def __init__(self, module):
        self.module = module


class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.block = Block()
        self.conv = nn.Conv2d(3, 3, 1)


def assert_tensor_equal(tensor_a, tensor_b):
    assert tensor_a.eq(tensor_b).all()


def test_get_state_dict():
    model = Model()
    state_dict = get_state_dict(model)
    assert isinstance(state_dict, OrderedDict)
    assert set(state_dict.keys()) == set(
        ['block.conv.weight', 'block.conv.bias', 'conv.weight', 'conv.bias'])

    assert_tensor_equal(state_dict['block.conv.weight'],
                        model.block.conv.weight.data)
    assert_tensor_equal(state_dict['block.conv.bias'],
                        model.block.conv.bias.data)
    assert_tensor_equal(state_dict['conv.weight'], model.conv.weight.data)
    assert_tensor_equal(state_dict['conv.bias'], model.conv.bias.data)

    ddp_wrapped_model = DDPWrapper(model)
    state_dict = get_state_dict(ddp_wrapped_model)
    assert isinstance(state_dict, OrderedDict)
    assert set(state_dict.keys()) == set(
        ['block.conv.weight', 'block.conv.bias', 'conv.weight', 'conv.bias'])
    assert_tensor_equal(state_dict['block.conv.weight'],
                        ddp_wrapped_model.module.block.conv.weight.data)
    assert_tensor_equal(state_dict['block.conv.bias'],
                        ddp_wrapped_model.module.block.conv.bias.data)
    assert_tensor_equal(state_dict['conv.weight'],
                        ddp_wrapped_model.module.conv.weight.data)
    assert_tensor_equal(state_dict['conv.bias'],
                        ddp_wrapped_model.module.conv.bias.data)

    # wrapped inner module
    for name, module in ddp_wrapped_model.module._modules.items():
        module = DataParallel(module)
        ddp_wrapped_model.module._modules[name] = module
    state_dict = get_state_dict(ddp_wrapped_model)
    assert isinstance(state_dict, OrderedDict)
    assert set(state_dict.keys()) == set(
        ['block.conv.weight', 'block.conv.bias', 'conv.weight', 'conv.bias'])
    assert_tensor_equal(state_dict['block.conv.weight'],
                        ddp_wrapped_model.module.block.module.conv.weight.data)
    assert_tensor_equal(state_dict['block.conv.bias'],
                        ddp_wrapped_model.module.block.module.conv.bias.data)
    assert_tensor_equal(state_dict['conv.weight'],
                        ddp_wrapped_model.module.conv.module.weight.data)
    assert_tensor_equal(state_dict['conv.bias'],
                        ddp_wrapped_model.module.conv.module.bias.data)
