# Copyright (c) Open-MMLab. All rights reserved.
import os
import os.path as osp
from unittest.mock import patch

import pytest

import mmcv
from mmcv.runner.checkpoint import (DEFAULT_CACHE_DIR, ENV_MMCV_HOME,
                                    ENV_XDG_CACHE_HOME, _get_mmcv_home,
                                    _load_checkpoint, get_external_models)


@patch('mmcv.__path__', [osp.join(osp.dirname(__file__), 'data/')])
def test_set_mmcv_home():
    os.environ.pop(ENV_MMCV_HOME, None)
    mmcv_home = osp.join(osp.dirname(__file__), 'data/urls/a/')
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
        osp.join(mmcv.__path__[0], 'urls/open_mmlab.json'))


@patch('mmcv.__path__', [osp.join(osp.dirname(__file__), 'data/')])
def test_get_external_models():
    os.environ.pop(ENV_MMCV_HOME, None)
    mmcv_home = osp.join(osp.dirname(__file__), 'data/urls/a/')
    os.environ[ENV_MMCV_HOME] = mmcv_home
    ext_urls = get_external_models()
    assert ext_urls == {
        'train': 'https://localhost/train.pth',
        'test': 'https://localhost/test1.pth',
        'test_x': 'https://localhost/test_x.pth'
    }

    # external urls should not have overlapped keys
    with pytest.raises(
            AssertionError,
            match='External urls should not have overlapped keys: '
            "{'test'}"):
        os.environ.pop(ENV_MMCV_HOME, None)
        mmcv_home = osp.join(osp.dirname(__file__), 'data/urls/b/')
        os.environ[ENV_MMCV_HOME] = mmcv_home
        get_external_models()


def load_url(url, model_dir=None):
    return url, model_dir


@patch('mmcv.__path__', [osp.join(osp.dirname(__file__), 'data/')])
@patch('mmcv.runner.checkpoint.load_url_dist', load_url)
def test_load_external_url():
    # test modelzoo://
    url, model_dir = _load_checkpoint('modelzoo://resnet50')
    assert url == 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    assert model_dir is None

    # test torchvision://
    url, model_dir = _load_checkpoint('torchvision://resnet50')
    assert url == 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    assert model_dir is None

    # test open-mmlab://
    os.environ.pop(ENV_MMCV_HOME, None)
    os.environ.pop(ENV_XDG_CACHE_HOME, None)
    url, model_dir = _load_checkpoint('open-mmlab://train')
    assert url == 'https://localhost/train.pth'
    assert model_dir == os.path.expanduser(
        os.path.join(DEFAULT_CACHE_DIR, 'mmcv'))

    # test local://
    os.environ.pop(ENV_MMCV_HOME, None)
    mmcv_home = osp.join(osp.dirname(__file__), 'data/urls/c/')
    os.environ[ENV_MMCV_HOME] = mmcv_home
    url, model_dir = _load_checkpoint('local://test')
    assert url == 'test.pth'
    assert model_dir == mmcv_home
    url, model_dir = _load_checkpoint('local://test1')
    assert url == 'test1/test1.pth'
    assert model_dir == osp.join(mmcv_home, 'test1')

    # test http:// https://
    url, model_dir = _load_checkpoint('http://localhost/train.pth')
    assert url == 'http://localhost/train.pth'
    assert model_dir is None
    url, model_dir = _load_checkpoint('https://localhost/train.pth')
    assert url == 'https://localhost/train.pth'
    assert model_dir is None
