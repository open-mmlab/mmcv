# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from unittest.mock import patch

import pytest
import torchvision

import mmcv
from mmcv.runner.checkpoint import (DEFAULT_CACHE_DIR, ENV_MMCV_HOME,
                                    ENV_XDG_CACHE_HOME, _get_mmcv_home,
                                    _load_checkpoint,
                                    get_deprecated_model_names,
                                    get_external_models)
from mmcv.utils import digit_version


@patch('mmcv.__path__', [osp.join(osp.dirname(__file__), 'data/')])
def test_set_mmcv_home():
    os.environ.pop(ENV_MMCV_HOME, None)
    mmcv_home = osp.join(osp.dirname(__file__), 'data/model_zoo/mmcv_home/')
    os.environ[ENV_MMCV_HOME] = mmcv_home
    assert _get_mmcv_home() == mmcv_home


@patch('mmcv.__path__', [osp.join(osp.dirname(__file__), 'data/')])
def test_default_mmcv_home():
    os.environ.pop(ENV_MMCV_HOME, None)
    os.environ.pop(ENV_XDG_CACHE_HOME, None)
    assert _get_mmcv_home() == os.path.expanduser(
        os.path.join(DEFAULT_CACHE_DIR, 'mmcv'))
    model_urls = get_external_models()
    assert model_urls == mmcv.load(
        osp.join(mmcv.__path__[0], 'model_zoo/open_mmlab.json'))


@patch('mmcv.__path__', [osp.join(osp.dirname(__file__), 'data/')])
def test_get_external_models():
    os.environ.pop(ENV_MMCV_HOME, None)
    mmcv_home = osp.join(osp.dirname(__file__), 'data/model_zoo/mmcv_home/')
    os.environ[ENV_MMCV_HOME] = mmcv_home
    ext_urls = get_external_models()
    assert ext_urls == {
        'train': 'https://localhost/train.pth',
        'test': 'test.pth',
        'val': 'val.pth',
        'train_empty': 'train.pth'
    }


@patch('mmcv.__path__', [osp.join(osp.dirname(__file__), 'data/')])
def test_get_deprecated_models():
    os.environ.pop(ENV_MMCV_HOME, None)
    mmcv_home = osp.join(osp.dirname(__file__), 'data/model_zoo/mmcv_home/')
    os.environ[ENV_MMCV_HOME] = mmcv_home
    dep_urls = get_deprecated_model_names()
    assert dep_urls == {
        'train_old': 'train',
        'test_old': 'test',
    }


def load_from_http(url, map_location=None):
    return 'url:' + url


def load_url(url, map_location=None, model_dir=None):
    return load_from_http(url)


def load(filepath, map_location=None):
    return 'local:' + filepath


@patch('mmcv.__path__', [osp.join(osp.dirname(__file__), 'data/')])
@patch('mmcv.runner.checkpoint.load_from_http', load_from_http)
@patch('mmcv.runner.checkpoint.load_url', load_url)
@patch('torch.load', load)
def test_load_external_url():
    # test modelzoo://
    torchvision_version = torchvision.__version__
    if digit_version(torchvision_version) < digit_version('0.10.0a0'):
        assert (_load_checkpoint('modelzoo://resnet50') ==
                'url:https://download.pytorch.org/models/resnet50-19c8e'
                '357.pth')
        assert (_load_checkpoint('torchvision://resnet50') ==
                'url:https://download.pytorch.org/models/resnet50-19c8e'
                '357.pth')
    else:
        assert (_load_checkpoint('modelzoo://resnet50') ==
                'url:https://download.pytorch.org/models/resnet50-0676b'
                'a61.pth')
        assert (_load_checkpoint('torchvision://resnet50') ==
                'url:https://download.pytorch.org/models/resnet50-0676b'
                'a61.pth')

    if digit_version(torchvision_version) >= digit_version('0.13.0a0'):
        # Test load new format torchvision models.
        assert (
            _load_checkpoint('torchvision://resnet50.imagenet1k_v1') ==
            'url:https://download.pytorch.org/models/resnet50-0676ba61.pth')

        assert (
            _load_checkpoint('torchvision://ResNet50_Weights.IMAGENET1K_V1') ==
            'url:https://download.pytorch.org/models/resnet50-0676ba61.pth')

        _load_checkpoint('torchvision://resnet50.default')

    # test open-mmlab:// with default MMCV_HOME
    os.environ.pop(ENV_MMCV_HOME, None)
    os.environ.pop(ENV_XDG_CACHE_HOME, None)
    url = _load_checkpoint('open-mmlab://train')
    assert url == 'url:https://localhost/train.pth'

    # test open-mmlab:// with deprecated model name
    os.environ.pop(ENV_MMCV_HOME, None)
    os.environ.pop(ENV_XDG_CACHE_HOME, None)
    with pytest.warns(
            Warning,
            match='open-mmlab://train_old is deprecated in favor of '
            'open-mmlab://train'):
        url = _load_checkpoint('open-mmlab://train_old')
        assert url == 'url:https://localhost/train.pth'

    # test openmmlab:// with deprecated model name
    os.environ.pop(ENV_MMCV_HOME, None)
    os.environ.pop(ENV_XDG_CACHE_HOME, None)
    with pytest.warns(
            Warning,
            match='openmmlab://train_old is deprecated in favor of '
            'openmmlab://train'):
        url = _load_checkpoint('openmmlab://train_old')
        assert url == 'url:https://localhost/train.pth'

    # test open-mmlab:// with user-defined MMCV_HOME
    os.environ.pop(ENV_MMCV_HOME, None)
    mmcv_home = osp.join(osp.dirname(__file__), 'data/model_zoo/mmcv_home')
    os.environ[ENV_MMCV_HOME] = mmcv_home
    url = _load_checkpoint('open-mmlab://train')
    assert url == 'url:https://localhost/train.pth'
    with pytest.raises(FileNotFoundError, match='train.pth can not be found.'):
        _load_checkpoint('open-mmlab://train_empty')
    url = _load_checkpoint('open-mmlab://test')
    assert url == f'local:{osp.join(_get_mmcv_home(), "test.pth")}'
    url = _load_checkpoint('open-mmlab://val')
    assert url == f'local:{osp.join(_get_mmcv_home(), "val.pth")}'

    # test http:// https://
    url = _load_checkpoint('http://localhost/train.pth')
    assert url == 'url:http://localhost/train.pth'

    # test local file
    with pytest.raises(FileNotFoundError, match='train.pth can not be found.'):
        _load_checkpoint('train.pth')
    url = _load_checkpoint(osp.join(_get_mmcv_home(), 'test.pth'))
    assert url == f'local:{osp.join(_get_mmcv_home(), "test.pth")}'
