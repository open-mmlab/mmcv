import sys
from collections import OrderedDict
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

from mmcv.parallel.registry import MODULE_WRAPPERS
from mmcv.runner.checkpoint import (_load_checkpoint_with_prefix,
                                    get_state_dict, load_pavimodel_dist)


@MODULE_WRAPPERS.register_module()
class DDPWrapper(object):

    def __init__(self, module):
        self.module = module


class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.norm = nn.BatchNorm2d(3)


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.block = Block()
        self.conv = nn.Conv2d(3, 3, 1)


class Mockpavimodel(object):

    def __init__(self, name='fakename'):
        self.name = name

    def download(self, file):
        pass


def assert_tensor_equal(tensor_a, tensor_b):
    assert tensor_a.eq(tensor_b).all()


def test_get_state_dict():
    if torch.__version__ == 'parrots':
        state_dict_keys = set([
            'block.conv.weight', 'block.conv.bias', 'block.norm.weight',
            'block.norm.bias', 'block.norm.running_mean',
            'block.norm.running_var', 'conv.weight', 'conv.bias'
        ])
    else:
        state_dict_keys = set([
            'block.conv.weight', 'block.conv.bias', 'block.norm.weight',
            'block.norm.bias', 'block.norm.running_mean',
            'block.norm.running_var', 'block.norm.num_batches_tracked',
            'conv.weight', 'conv.bias'
        ])

    model = Model()
    state_dict = get_state_dict(model)
    assert isinstance(state_dict, OrderedDict)
    assert set(state_dict.keys()) == state_dict_keys

    assert_tensor_equal(state_dict['block.conv.weight'],
                        model.block.conv.weight)
    assert_tensor_equal(state_dict['block.conv.bias'], model.block.conv.bias)
    assert_tensor_equal(state_dict['block.norm.weight'],
                        model.block.norm.weight)
    assert_tensor_equal(state_dict['block.norm.bias'], model.block.norm.bias)
    assert_tensor_equal(state_dict['block.norm.running_mean'],
                        model.block.norm.running_mean)
    assert_tensor_equal(state_dict['block.norm.running_var'],
                        model.block.norm.running_var)
    if torch.__version__ != 'parrots':
        assert_tensor_equal(state_dict['block.norm.num_batches_tracked'],
                            model.block.norm.num_batches_tracked)
    assert_tensor_equal(state_dict['conv.weight'], model.conv.weight)
    assert_tensor_equal(state_dict['conv.bias'], model.conv.bias)

    wrapped_model = DDPWrapper(model)
    state_dict = get_state_dict(wrapped_model)
    assert isinstance(state_dict, OrderedDict)
    assert set(state_dict.keys()) == state_dict_keys
    assert_tensor_equal(state_dict['block.conv.weight'],
                        wrapped_model.module.block.conv.weight)
    assert_tensor_equal(state_dict['block.conv.bias'],
                        wrapped_model.module.block.conv.bias)
    assert_tensor_equal(state_dict['block.norm.weight'],
                        wrapped_model.module.block.norm.weight)
    assert_tensor_equal(state_dict['block.norm.bias'],
                        wrapped_model.module.block.norm.bias)
    assert_tensor_equal(state_dict['block.norm.running_mean'],
                        wrapped_model.module.block.norm.running_mean)
    assert_tensor_equal(state_dict['block.norm.running_var'],
                        wrapped_model.module.block.norm.running_var)
    if torch.__version__ != 'parrots':
        assert_tensor_equal(
            state_dict['block.norm.num_batches_tracked'],
            wrapped_model.module.block.norm.num_batches_tracked)
    assert_tensor_equal(state_dict['conv.weight'],
                        wrapped_model.module.conv.weight)
    assert_tensor_equal(state_dict['conv.bias'],
                        wrapped_model.module.conv.bias)

    # wrapped inner module
    for name, module in wrapped_model.module._modules.items():
        module = DataParallel(module)
        wrapped_model.module._modules[name] = module
    state_dict = get_state_dict(wrapped_model)
    assert isinstance(state_dict, OrderedDict)
    assert set(state_dict.keys()) == state_dict_keys
    assert_tensor_equal(state_dict['block.conv.weight'],
                        wrapped_model.module.block.module.conv.weight)
    assert_tensor_equal(state_dict['block.conv.bias'],
                        wrapped_model.module.block.module.conv.bias)
    assert_tensor_equal(state_dict['block.norm.weight'],
                        wrapped_model.module.block.module.norm.weight)
    assert_tensor_equal(state_dict['block.norm.bias'],
                        wrapped_model.module.block.module.norm.bias)
    assert_tensor_equal(state_dict['block.norm.running_mean'],
                        wrapped_model.module.block.module.norm.running_mean)
    assert_tensor_equal(state_dict['block.norm.running_var'],
                        wrapped_model.module.block.module.norm.running_var)
    if torch.__version__ != 'parrots':
        assert_tensor_equal(
            state_dict['block.norm.num_batches_tracked'],
            wrapped_model.module.block.module.norm.num_batches_tracked)
    assert_tensor_equal(state_dict['conv.weight'],
                        wrapped_model.module.conv.module.weight)
    assert_tensor_equal(state_dict['conv.bias'],
                        wrapped_model.module.conv.module.bias)


def test_load_pavimodel_dist():

    sys.modules['pavi'] = MagicMock()
    sys.modules['pavi.modelcloud'] = MagicMock()
    pavimodel = Mockpavimodel()
    import pavi
    pavi.modelcloud.get = MagicMock(return_value=pavimodel)
    with pytest.raises(FileNotFoundError):
        # there is not such checkpoint for us to load
        _ = load_pavimodel_dist('MyPaviFolder/checkpoint.pth')


def test_load_checkpoint_with_prefix():

    class FooModule(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 2)
            self.conv2d = nn.Conv2d(3, 1, 3)
            self.conv2d_2 = nn.Conv2d(3, 2, 3)

    model = FooModule()
    nn.init.constant_(model.linear.weight, 1)
    nn.init.constant_(model.linear.bias, 2)
    nn.init.constant_(model.conv2d.weight, 3)
    nn.init.constant_(model.conv2d.bias, 4)
    nn.init.constant_(model.conv2d_2.weight, 5)
    nn.init.constant_(model.conv2d_2.bias, 6)

    with TemporaryDirectory():
        torch.save(model.state_dict(), 'model.pth')
        prefix = 'conv2d'
        state_dict = _load_checkpoint_with_prefix(prefix, 'model.pth')
        assert torch.equal(model.conv2d.state_dict()['weight'],
                           state_dict['weight'])
        assert torch.equal(model.conv2d.state_dict()['bias'],
                           state_dict['bias'])

        prefix = 'back'
        with pytest.raises(AssertionError):
            _load_checkpoint_with_prefix(prefix, 'model.pth')


def test_load_classes_name():
    import os

    import tempfile

    from mmcv.runner import load_checkpoint, save_checkpoint
    checkpoint_path = os.path.join(tempfile.gettempdir(), 'checkpoint.pth')
    model = Model()
    save_checkpoint(model, checkpoint_path)
    checkpoint = load_checkpoint(model, checkpoint_path)
    assert 'meta' in checkpoint and 'CLASSES' not in checkpoint['meta']

    model.CLASSES = ('class1', 'class2')
    save_checkpoint(model, checkpoint_path)
    checkpoint = load_checkpoint(model, checkpoint_path)
    assert 'meta' in checkpoint and 'CLASSES' in checkpoint['meta']
    assert checkpoint['meta']['CLASSES'] == ('class1', 'class2')

    model = Model()
    wrapped_model = DDPWrapper(model)
    save_checkpoint(wrapped_model, checkpoint_path)
    checkpoint = load_checkpoint(wrapped_model, checkpoint_path)
    assert 'meta' in checkpoint and 'CLASSES' not in checkpoint['meta']

    wrapped_model.module.CLASSES = ('class1', 'class2')
    save_checkpoint(wrapped_model, checkpoint_path)
    checkpoint = load_checkpoint(wrapped_model, checkpoint_path)
    assert 'meta' in checkpoint and 'CLASSES' in checkpoint['meta']
    assert checkpoint['meta']['CLASSES'] == ('class1', 'class2')

    # remove the temp file
    os.remove(checkpoint_path)
