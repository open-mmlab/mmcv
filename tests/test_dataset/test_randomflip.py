import copy
import os.path as osp

import numpy as np
import pytest

import mmcv
from mmcv.datasets.pipelines.transforms import RandomFlip


class TestRandomFlip:

    @classmethod
    def setup_class(cls):
        cls.color_img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')

        bbox_det = np.array([[10, 20, 40, 50], [20, 20, 70, 100]],
                            dtype=np.int32)
        cls.results_bbox = dict(
            img_shape=(256, 256), bbox_fields=['bbox_det'], bbox_det=bbox_det)

    def test_randomflip(self):
        # test assertion for invalid prob
        with pytest.raises(AssertionError):
            RandomFlip(prob=1.5)

        with pytest.raises(ValueError):
            RandomFlip(prob='str')

        # test assertion for 0 <= sum(prob) <= 1
        with pytest.raises(AssertionError):
            RandomFlip(prob=[0.7, 0.8], direction=['horizontal', 'vertical'])

        # test assertion for mismatch between number of prob and direction
        with pytest.raises(AssertionError):
            RandomFlip(prob=[0.7, 0.1], direction=['vertical'])

        # test assertion for invalid direction
        with pytest.raises(AssertionError):
            RandomFlip(prob=[0.7, 0.1], direction=['horizontal', 'vertica'])

        with pytest.raises(ValueError):
            RandomFlip(prob=[0.7, 0.1], direction=1)

        flip_module = RandomFlip(prob=1.)

        results = dict()
        img = self.color_img
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

        flip_module = RandomFlip(prob=None)

        results = dict()
        img = self.color_img
        original_img = copy.deepcopy(img)
        results['img'] = img
        results['img2'] = copy.deepcopy(img)
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['img_fields'] = ['img', 'img2']
        results['flip'] = True
        results['flip_direction'] = 'horizontal'

        results = flip_module(results)
        assert np.equal(results['img'], results['img2']).all()

        results = flip_module(results)
        assert np.equal(results['img'], results['img2']).all()
        assert np.equal(original_img, results['img']).all()

        # test prob is float, direction is list
        flip_module = RandomFlip(
            prob=0.9, direction=['horizontal', 'vertical', 'diagonal'])

        results = dict()
        img = self.color_img
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
        img = self.color_img
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

        # test bbox flip
        flip_module = RandomFlip(prob=1., flip_fields=['bbox'])
        results = copy.deepcopy(self.results_bbox)

        results = flip_module(results)
        bbox_horizontal = np.array([[246, 20, 216, 50], [236, 20, 186, 100]],
                                   dtype=np.int32)
        np.array_equal(results['bbox_det'], bbox_horizontal)

        flip_module = RandomFlip(
            prob=1., direction='vertical', flip_fields=['bbox'])
        results = copy.deepcopy(self.results_bbox)

        results = flip_module(results)
        bbox_vertical = np.array([[10, 236, 40, 210], [20, 236, 70, 156]],
                                 dtype=np.int32)
        np.array_equal(results['bbox_det'], bbox_vertical)

        flip_module = RandomFlip(
            prob=1., direction='diagonal', flip_fields=['bbox'])
        results = copy.deepcopy(self.results_bbox)

        results = flip_module(results)
        bbox_diagonal = np.array([[246, 236, 216, 210], [236, 236, 186, 156]],
                                 dtype=np.int32)
        np.array_equal(results['bbox_det'], bbox_diagonal)

        # test the name not in the pre-defined list
        flip_module = RandomFlip(prob=1., flip_fields=['img1', 'img2'])

        results = dict()
        img = self.color_img
        original_img = copy.deepcopy(img)
        results['img1'] = img
        results['img2'] = copy.deepcopy(img)
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['img_fields'] = ['img1', 'img2']

        results = flip_module(results)
        assert np.equal(results['img1'], results['img2']).all()

        if results['flip']:
            assert np.array_equal(
                mmcv.imflip(original_img, results['flip_direction']),
                results['img1'])
        else:
            assert np.array_equal(original_img, results['img'])

        assert str(
            flip_module) == flip_module.__class__.__name__ + f'(prob={1.})'
