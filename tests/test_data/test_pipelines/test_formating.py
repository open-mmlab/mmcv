import os.path as osp

import numpy as np
import pytest

from mmcv.datasets.builder import PIPELINES
from mmcv.image.io import imread
from mmcv.parallel import DataContainer
from mmcv.utils.registry import build_from_cfg


def test_to_data_container():
    results = dict()
    img = imread(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
    bboxes = np.random.uniform(0, 100, size=(5, 4))
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['gt_bboxes'] = bboxes

    with pytest.raises(KeyError):
        transform_cfg = dict(
            type='ToDataContainer',
            fields=(dict(key='img', stack=True), dict(key='abc')))
        transform = build_from_cfg(transform_cfg, PIPELINES)
        results = transform(results)

    transform_cfg = dict(
        type='ToDataContainer',
        fields=(dict(key='img', stack=True), dict(key='gt_bboxes')))
    transform = build_from_cfg(transform_cfg, PIPELINES)

    results['img_fields'] = ['img']
    results = transform(results)
    assert isinstance(results['img'], DataContainer)
    assert isinstance(results['gt_bboxes'], DataContainer)
