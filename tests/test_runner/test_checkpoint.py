import sys
from collections import OrderedDict
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

from mmcv.parallel.registry import MODULE_WRAPPERS
from mmcv.runner.checkpoint import get_state_dict, load_from_pavi


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
        _ = load_from_pavi('MyPaviFolder/checkpoint.pth')


def test_load_classes_name():
    from mmcv.runner import load_checkpoint, save_checkpoint
    import tempfile
    import os
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


def test_checkpoint_loader():
    from mmcv.runner import _load_checkpoint, save_checkpoint, CheckpointLoader
    import tempfile
    import os
    checkpoint_path = os.path.join(tempfile.gettempdir(), 'checkpoint.pth')
    model = Model()
    save_checkpoint(model, checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    assert 'meta' in checkpoint and 'CLASSES' not in checkpoint['meta']
    # remove the temp file
    os.remove(checkpoint_path)

    filenames = [
        'http://xx.xx/xx.pth', 'https://xx.xx/xx.pth',
        'modelzoo://xx.xx/xx.pth', 'torchvision://xx.xx/xx.pth',
        'open-mmlab://xx.xx/xx.pth', 'mmcls://xx.xx/xx.pth',
        'pavi://xx.xx/xx.pth', 's3://xx.xx/xx.pth', 'ss3://xx.xx/xx.pth'
    ]
    fn_names = [
        'load_from_http', 'load_from_http', 'load_from_torchvision',
        'load_from_torchvision', 'load_from_openmmlab', 'load_from_mmcls',
        'load_from_pavi', 'load_from_ceph', 'load_from_local'
    ]

    for filename, fn_name in zip(filenames, fn_names):
        loader = CheckpointLoader._get_checkpoint_loader(filename)
        assert loader.__name__ == fn_name

    @CheckpointLoader.register_scheme(prefixes='ftp://')
    def load_from_ftp(filename, map_location):
        return dict(filename=filename)

    # test register_loader
    filename = 'ftp://xx.xx/xx.pth'
    loader = CheckpointLoader._get_checkpoint_loader(filename)
    assert loader.__name__ == 'load_from_ftp'

    def load_from_ftp1(filename, map_location):
        return dict(filename=filename)

    # test duplicate registered error
    with pytest.raises(KeyError):
        CheckpointLoader.register_scheme('ftp://', load_from_ftp1)

    # test force param
    CheckpointLoader.register_scheme('ftp://', load_from_ftp1, force=True)
    checkpoint = CheckpointLoader.load_checkpoint(filename)
    assert checkpoint['filename'] == filename

    # test print function name
    loader = CheckpointLoader._get_checkpoint_loader(filename)
    assert loader.__name__ == 'load_from_ftp1'

    # test sort
    @CheckpointLoader.register_scheme(prefixes='a/b')
    def load_from_ab(filename, map_location):
        return dict(filename=filename)

    @CheckpointLoader.register_scheme(prefixes='a/b/c')
    def load_from_abc(filename, map_location):
        return dict(filename=filename)

    filename = 'a/b/c/d'
    loader = CheckpointLoader._get_checkpoint_loader(filename)
    assert loader.__name__ == 'load_from_abc'
