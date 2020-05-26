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


def load_url_dist(url):
    return 'url:' + url


def load(filepath, map_location=None):
    return 'local:' + filepath


@patch('mmcv.__path__', [osp.join(osp.dirname(__file__), 'data/')])
@patch('mmcv.runner.checkpoint.load_url_dist', load_url_dist)
@patch('torch.load', load)
def test_load_external_url():
    # test modelzoo://
    url = _load_checkpoint('modelzoo://resnet50')
    assert url == 'url:https://download.pytorch.org/models/resnet50-19c8e357' \
                  '.pth'

    # test torchvision://
    url = _load_checkpoint('torchvision://resnet50')
    assert url == 'url:https://download.pytorch.org/models/resnet50-19c8e357' \
                  '.pth'

    # test open-mmlab:// with default MMCV_HOME
    os.environ.pop(ENV_MMCV_HOME, None)
    os.environ.pop(ENV_XDG_CACHE_HOME, None)
    url = _load_checkpoint('open-mmlab://train')
    assert url == 'url:https://localhost/train.pth'

    # test open-mmlab:// with user-defined MMCV_HOME
    os.environ.pop(ENV_MMCV_HOME, None)
    mmcv_home = osp.join(osp.dirname(__file__), 'data/model_zoo/mmcv_home')
    os.environ[ENV_MMCV_HOME] = mmcv_home
    url = _load_checkpoint('open-mmlab://train')
    assert url == 'url:https://localhost/train.pth'
    with pytest.raises(IOError, match='train.pth is not a checkpoint ' 'file'):
        _load_checkpoint('open-mmlab://train_empty')
    url = _load_checkpoint('open-mmlab://test')
    assert url == f'local:{osp.join(_get_mmcv_home(), "test.pth")}'
    url = _load_checkpoint('open-mmlab://val')
    assert url == f'local:{osp.join(_get_mmcv_home(), "val.pth")}'

    # test http:// https://
    url = _load_checkpoint('http://localhost/train.pth')
    assert url == 'url:http://localhost/train.pth'

    # test local file
    with pytest.raises(IOError, match='train.pth is not a checkpoint ' 'file'):
        _load_checkpoint('train.pth')
    url = _load_checkpoint(osp.join(_get_mmcv_home(), 'test.pth'))
    assert url == f'local:{osp.join(_get_mmcv_home(), "test.pth")}'
