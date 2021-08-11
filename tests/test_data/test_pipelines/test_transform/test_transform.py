import copy
import os.path as osp

import numpy as np

import mmcv
from mmcv.datasets.builder import PIPELINES
from mmcv.utils import build_from_cfg


def test_normalize():
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
    transform_cfg = dict(type='Normalize', **img_norm_cfg)
    transform = build_from_cfg(transform_cfg, PIPELINES)
    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['img_fields'] = ['img', 'img2']

    results = transform(results)
    assert np.equal(results['img'], results['img2']).all()

    mean = np.array(img_norm_cfg['mean'])
    std = np.array(img_norm_cfg['std'])
    converted_img = (original_img[..., ::-1] - mean) / std
    assert np.allclose(results['img'], converted_img)
