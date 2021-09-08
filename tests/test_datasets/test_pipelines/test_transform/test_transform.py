import copy
import os.path as osp

import numpy as np
import pytest

from mmcv.datasets.builder import PIPELINES
from mmcv.image.io import imread
from mmcv.utils.registry import build_from_cfg


@pytest.mark.parametrize('is_single_channel,to_rgb', [(True, False),
                                                      (False, True),
                                                      (False, False)])
def test_normalize(is_single_channel, to_rgb):
    results = dict()
    if is_single_channel:
        img_norm_cfg = dict(mean=123.675, std=58.395, to_rgb=to_rgb)
        original_img = imread(
            osp.join(osp.dirname(__file__), '../../../data/grayscale.jpg'),
            'grayscale')
    else:
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=to_rgb)
        original_img = imread(
            osp.join(osp.dirname(__file__), '../../../data/color.jpg'),
            'color')
    results['img'] = copy.deepcopy(original_img)
    results['img2'] = copy.deepcopy(original_img)

    transform_cfg = dict(type='Normalize', **img_norm_cfg)
    transform = build_from_cfg(transform_cfg, PIPELINES)

    with pytest.raises(AssertionError):
        # Required key of results is 'img_fields'
        results = transform(results)

    results['img_fields'] = ['img', 'img2']
    results = transform(results)
    assert np.equal(results['img'], results['img2']).all()

    mean = np.array(img_norm_cfg['mean'])
    std = np.array(img_norm_cfg['std'])
    if to_rgb:
        original_img = original_img[..., ::-1]
    converted_img = (original_img.astype(np.float32) - mean) / std
    assert np.allclose(results['img'], converted_img)
