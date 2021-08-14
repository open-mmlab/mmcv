import copy
import os.path as osp

import numpy as np
import pytest

import mmcv
from mmcv.datasets.pipelines.transforms import RandomFlip


class TestRandomFlip:

    def test_randomflip(self):
        # test assertion for invalid prob
        with pytest.raises(AssertionError):
            RandomFlip(prob=1.5)
        # test assertion for 0 <= sum(prob) <= 1
        with pytest.raises(AssertionError):
            RandomFlip(prob=[0.7, 0.8], direction=['horizontal', 'vertical'])

        # test assertion for mismatch between number of prob and direction
        with pytest.raises(AssertionError):
            RandomFlip(prob=[0.7, 0.1], direction=['vertical'])

        # test assertion for invalid direction
        with pytest.raises(AssertionError):
            RandomFlip(prob=[0.7, 0.1], direction=['horizontal', 'vertica'])

        flip_module = RandomFlip(prob=1.)

        results = dict()
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
        original_img = copy.deepcopy(img)
        results['img'] = img
        results['img2'] = copy.deepcopy(img)
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['img_fields'] = ['img', 'img2']

        results = flip_module(results)
        assert np.equal(results['img'], results['img2']).all()

        results = flip_module(results)
        assert np.equal(results['img'], results['img2']).all()
        assert np.equal(original_img, results['img']).all()

        # test prob is float, direction is list
        flip_module = RandomFlip(
            prob=0.9, direction=['horizontal', 'vertical', 'diagonal'])

        results = dict()
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
        original_img = copy.deepcopy(img)
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['img_fields'] = ['img']
        results = flip_module(results)
        if results['flip']:
            assert np.array_equal(
                mmcv.imflip(original_img, results['flip_direction']),
                results['img'])
        else:
            assert np.array_equal(original_img, results['img'])

        # test prob is list, direction is list
        flip_module = RandomFlip(
            prob=[0.3, 0.3, 0.2],
            direction=['horizontal', 'vertical', 'diagonal'])

        results = dict()
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
        original_img = copy.deepcopy(img)
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['img_fields'] = ['img']
        results = flip_module(results)
        if results['flip']:
            assert np.array_equal(
                mmcv.imflip(original_img, results['flip_direction']),
                results['img'])
        else:
            assert np.array_equal(original_img, results['img'])
