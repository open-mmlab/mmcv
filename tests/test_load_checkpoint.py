# Copyright (c) Open-MMLab. All rights reserved.
import os
import os.path as osp
import warnings

import pytest

import mmcv
from mmcv.runner.checkpoint import ENV_MMCV_HOME, get_openmmlab_models


def test_set_mmcv_home():
    with warnings.catch_warnings(record=True) as w:
        os.environ[ENV_MMCV_HOME] = osp.join(osp.dirname(__file__), 'data')
        warnings.simplefilter('always')
        model_urls = get_openmmlab_models()
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert str(w[0].message) == 'Using {} for downloading open-mmlab ' \
                                    'models.'.format(
            osp.abspath(osp.join(osp.dirname(__file__),
                                 'data/urls/open_mmlab.json')))
        assert model_urls == dict(test='https://localhost/test.pth')

    with pytest.raises(FileNotFoundError):
        os.environ[ENV_MMCV_HOME] = osp.join(osp.dirname(__file__), 'wrong')
        get_openmmlab_models()


def test_default_mmcv_home():
    os.environ.pop(ENV_MMCV_HOME)
    model_urls = get_openmmlab_models()
    assert model_urls == mmcv.load(
        osp.join(mmcv.__path__[0], 'urls/open_mmlab.json'))
