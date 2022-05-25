# Copyright (c) OpenMMLab. All rights reserved.
import sys
from collections import OrderedDict
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel

from mmcv.fileio.file_client import PetrelBackend
from mmcv.parallel.registry import MODULE_WRAPPERS
from mmcv.runner.checkpoint import (_load_checkpoint_with_prefix,
                                    get_state_dict, load_checkpoint,
                                    load_from_local, load_from_pavi,
                                    save_checkpoint)

sys.modules['petrel_client'] = MagicMock()
sys.modules['petrel_client.client'] = MagicMock()


@MODULE_WRAPPERS.register_module()
class DDPWrapper:

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


class Mockpavimodel:

    def __init__(self, name='fakename'):
        self.name = name

    def download(self, file):
        pass


def assert_tensor_equal(tensor_a, tensor_b):
    assert tensor_a.eq(tensor_b).all()


def test_get_state_dict():
    if torch.__version__ == 'parrots':
        state_dict_keys = {
            'block.conv.weight', 'block.conv.bias', 'block.norm.weight',
            'block.norm.bias', 'block.norm.running_mean',
            'block.norm.running_var', 'conv.weight', 'conv.bias'
        }
    else:
        state_dict_keys = {
            'block.conv.weight', 'block.conv.bias', 'block.norm.weight',
            'block.norm.bias', 'block.norm.running_mean',
            'block.norm.running_var', 'block.norm.num_batches_tracked',
            'conv.weight', 'conv.bias'
        }

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
    with pytest.raises(AssertionError):
        # test pavi prefix
        _ = load_from_pavi('MyPaviFolder/checkpoint.pth')

    with pytest.raises(FileNotFoundError):
        # there is not such checkpoint for us to load
        _ = load_from_pavi('pavi://checkpoint.pth')


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

        # test whether prefix is in pretrained model
        with pytest.raises(AssertionError):
            prefix = 'back'
            _load_checkpoint_with_prefix(prefix, 'model.pth')


def test_load_checkpoint():
    import os
    import re
    import tempfile

    class PrefixModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.backbone = Model()

    pmodel = PrefixModel()
    model = Model()
    checkpoint_path = os.path.join(tempfile.gettempdir(), 'checkpoint.pth')

    # add prefix
    torch.save(model.state_dict(), checkpoint_path)
    state_dict = load_checkpoint(
        pmodel, checkpoint_path, revise_keys=[(r'^', 'backbone.')])
    for key in pmodel.backbone.state_dict().keys():
        assert torch.equal(pmodel.backbone.state_dict()[key], state_dict[key])
    # strip prefix
    torch.save(pmodel.state_dict(), checkpoint_path)
    state_dict = load_checkpoint(
        model, checkpoint_path, revise_keys=[(r'^backbone\.', '')])

    for key in state_dict.keys():
        key_stripped = re.sub(r'^backbone\.', '', key)
        assert torch.equal(model.state_dict()[key_stripped], state_dict[key])
    os.remove(checkpoint_path)


def test_load_checkpoint_metadata():
    import os
    import tempfile

    from mmcv.runner import load_checkpoint, save_checkpoint

    class ModelV1(nn.Module):

        def __init__(self):
            super().__init__()
            self.block = Block()
            self.conv1 = nn.Conv2d(3, 3, 1)
            self.conv2 = nn.Conv2d(3, 3, 1)
            nn.init.normal_(self.conv1.weight)
            nn.init.normal_(self.conv2.weight)

    class ModelV2(nn.Module):
        _version = 2

        def __init__(self):
            super().__init__()
            self.block = Block()
            self.conv0 = nn.Conv2d(3, 3, 1)
            self.conv1 = nn.Conv2d(3, 3, 1)
            nn.init.normal_(self.conv0.weight)
            nn.init.normal_(self.conv1.weight)

        def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                  *args, **kwargs):
            """load checkpoints."""

            # Names of some parameters in has been changed.
            version = local_metadata.get('version', None)
            if version is None or version < 2:
                state_dict_keys = list(state_dict.keys())
                convert_map = {'conv1': 'conv0', 'conv2': 'conv1'}
                for k in state_dict_keys:
                    for ori_str, new_str in convert_map.items():
                        if k.startswith(prefix + ori_str):
                            new_key = k.replace(ori_str, new_str)
                            state_dict[new_key] = state_dict[k]
                            del state_dict[k]

            super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                          *args, **kwargs)

    model_v1 = ModelV1()
    model_v1_conv0_weight = model_v1.conv1.weight.detach()
    model_v1_conv1_weight = model_v1.conv2.weight.detach()
    model_v2 = ModelV2()
    model_v2_conv0_weight = model_v2.conv0.weight.detach()
    model_v2_conv1_weight = model_v2.conv1.weight.detach()
    ckpt_v1_path = os.path.join(tempfile.gettempdir(), 'checkpoint_v1.pth')
    ckpt_v2_path = os.path.join(tempfile.gettempdir(), 'checkpoint_v2.pth')

    # Save checkpoint
    save_checkpoint(model_v1, ckpt_v1_path)
    save_checkpoint(model_v2, ckpt_v2_path)

    # test load v1 model
    load_checkpoint(model_v2, ckpt_v1_path)
    assert torch.allclose(model_v2.conv0.weight, model_v1_conv0_weight)
    assert torch.allclose(model_v2.conv1.weight, model_v1_conv1_weight)

    # test load v2 model
    load_checkpoint(model_v2, ckpt_v2_path)
    assert torch.allclose(model_v2.conv0.weight, model_v2_conv0_weight)
    assert torch.allclose(model_v2.conv1.weight, model_v2_conv1_weight)


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


def test_checkpoint_loader():
    import os
    import tempfile

    from mmcv.runner import CheckpointLoader, _load_checkpoint, save_checkpoint
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
        'open-mmlab://xx.xx/xx.pth', 'openmmlab://xx.xx/xx.pth',
        'mmcls://xx.xx/xx.pth', 'pavi://xx.xx/xx.pth', 's3://xx.xx/xx.pth',
        'ss3://xx.xx/xx.pth', ' s3://xx.xx/xx.pth',
        'open-mmlab:s3://xx.xx/xx.pth', 'openmmlab:s3://xx.xx/xx.pth',
        'openmmlabs3://xx.xx/xx.pth', ':s3://xx.xx/xx.path'
    ]
    fn_names = [
        'load_from_http', 'load_from_http', 'load_from_torchvision',
        'load_from_torchvision', 'load_from_openmmlab', 'load_from_openmmlab',
        'load_from_mmcls', 'load_from_pavi', 'load_from_ceph',
        'load_from_local', 'load_from_local', 'load_from_ceph',
        'load_from_ceph', 'load_from_local', 'load_from_local'
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


def test_save_checkpoint(tmp_path):
    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # meta is not a dict
    with pytest.raises(TypeError):
        save_checkpoint(model, '/path/of/your/filename', meta='invalid type')

    # 1. save to disk
    filename = str(tmp_path / 'checkpoint1.pth')
    save_checkpoint(model, filename)

    filename = str(tmp_path / 'checkpoint2.pth')
    save_checkpoint(model, filename, optimizer)

    filename = str(tmp_path / 'checkpoint3.pth')
    save_checkpoint(model, filename, meta={'test': 'test'})

    filename = str(tmp_path / 'checkpoint4.pth')
    save_checkpoint(model, filename, file_client_args={'backend': 'disk'})

    # 2. save to petrel oss
    with patch.object(PetrelBackend, 'put') as mock_method:
        filename = 's3://path/of/your/checkpoint1.pth'
        save_checkpoint(model, filename)
    mock_method.assert_called()

    with patch.object(PetrelBackend, 'put') as mock_method:
        filename = 's3://path//of/your/checkpoint2.pth'
        save_checkpoint(
            model, filename, file_client_args={'backend': 'petrel'})
    mock_method.assert_called()


def test_load_from_local():
    import os
    home_path = os.path.expanduser('~')
    checkpoint_path = os.path.join(
        home_path, 'dummy_checkpoint_used_to_test_load_from_local.pth')
    model = Model()
    save_checkpoint(model, checkpoint_path)
    checkpoint = load_from_local(
        '~/dummy_checkpoint_used_to_test_load_from_local.pth',
        map_location=None)
    assert_tensor_equal(checkpoint['state_dict']['block.conv.weight'],
                        model.block.conv.weight)
    os.remove(checkpoint_path)
