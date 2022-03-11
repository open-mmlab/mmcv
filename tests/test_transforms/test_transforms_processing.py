# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from unittest.mock import Mock

import numpy as np
import pytest

import mmcv
from mmcv.transforms import (TRANSFORMS, Normalize, Pad, RandomFlip,
                             RandomResize, Resize)
from mmcv.transforms.base import BaseTransform

try:
    import torch
except ModuleNotFoundError:
    torch = None
else:
    import torchvision

from numpy.testing import assert_array_almost_equal, assert_array_equal
from PIL import Image


class TestNormalize:

    def test_normalize(self):
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True)
        transform = Normalize(**img_norm_cfg)
        results = dict()
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
        original_img = copy.deepcopy(img)
        results['img'] = img
        results = transform(results)
        mean = np.array(img_norm_cfg['mean'])
        std = np.array(img_norm_cfg['std'])
        converted_img = (original_img[..., ::-1] - mean) / std
        assert np.allclose(results['img'], converted_img)

    def test_repr(self):
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True)
        transform = Normalize(**img_norm_cfg)
        assert repr(transform) == ('Normalize(mean=[123.675 116.28  103.53 ], '
                                   'std=[58.395 57.12  57.375], to_rgb=True)')


class TestResize:

    def test_resize(self):
        data_info = dict(
            img=np.random.random((1333, 800, 3)),
            gt_semantic_seg=np.random.random((1333, 800, 3)),
            gt_bboxes=np.array([[0, 0, 112, 112]]),
            gt_keypoints=np.array([[[20, 50, 1]]]))

        with pytest.raises(AssertionError):
            transform = Resize(scale=None, scale_factor=None)
        with pytest.raises(TypeError):
            transform = Resize(scale_factor=[])
        # test scale is int
        transform = Resize(scale=2000)
        results = transform(copy.deepcopy(data_info))
        assert results['img'].shape[:2] == (2000, 2000)
        assert results['scale_factor'] == (2000 / 800, 2000 / 1333)

        # test scale is tuple
        transform = Resize(scale=(2000, 2000))
        results = transform(copy.deepcopy(data_info))
        assert results['img'].shape[:2] == (2000, 2000)
        assert results['scale_factor'] == (2000 / 800, 2000 / 1333)

        # test scale_factor is float
        transform = Resize(scale_factor=2.0)
        results = transform(copy.deepcopy(data_info))
        assert results['img'].shape[:2] == (2666, 1600)
        assert results['scale_factor'] == (2.0, 2.0)

        # test scale_factor is tuple
        transform = Resize(scale_factor=(1.5, 2))
        results = transform(copy.deepcopy(data_info))
        assert results['img'].shape[:2] == (2666, 1200)
        assert results['scale_factor'] == (1.5, 2)

        # test keep_ratio is True
        transform = Resize(scale=(2000, 2000), keep_ratio=True)
        results = transform(copy.deepcopy(data_info))
        assert results['img'].shape[:2] == (2000, 1200)
        assert results['scale'] == (1200, 2000)
        assert results['scale_factor'] == (1200 / 800, 2000 / 1333)

        # test resize_bboxes/seg/kps
        transform = Resize(scale_factor=(1.5, 2))
        results = transform(copy.deepcopy(data_info))
        assert (results['gt_bboxes'] == np.array([[0, 0, 168, 224]])).all()
        assert (results['gt_keypoints'] == np.array([[[30, 100, 1]]])).all()
        assert results['gt_semantic_seg'].shape[:2] == (2666, 1200)

        # test clip_object_border = False
        data_info = dict(
            img=np.random.random((300, 400, 3)),
            gt_bboxes=np.array([[200, 150, 600, 450]]))
        transform = Resize(scale=(200, 150), clip_object_border=False)
        results = transform(data_info)
        assert (results['gt_bboxes'] == np.array([100, 75, 300, 225])).all()

    def test_repr(self):
        transform = Resize(scale=(2000, 2000), keep_ratio=True)
        assert repr(transform) == ('Resize(scale=(2000, 2000), '
                                   'scale_factor=None, keep_ratio=True, '
                                   'clip_object_border=True), backend=cv2), '
                                   'interpolation=bilinear)')


class TestPad:

    def test_pad(self):
        # test size and size_divisor are both set
        with pytest.raises(AssertionError):
            Pad(size=(10, 10), size_divisor=2)

        # test size and size_divisor are both None
        with pytest.raises(AssertionError):
            Pad(size=None, size_divisor=None)

        # test size and pad_to_square are both None
        with pytest.raises(AssertionError):
            Pad(size=(10, 10), pad_to_square=True)

        # test pad_val is not int or tuple
        with pytest.raises(AssertionError):
            Pad(size=(10, 10), pad_val=[])

        # test padding_mode is not 'constant', 'edge', 'reflect' or 'symmetric'
        with pytest.raises(AssertionError):
            Pad(size=(10, 10), padding_mode='edg')

        data_info = dict(
            img=np.random.random((1333, 800, 3)),
            gt_semantic_seg=np.random.random((1333, 800, 3)),
            gt_bboxes=np.array([[0, 0, 112, 112]]),
            gt_keypoints=np.array([[[20, 50, 1]]]))

        # test pad img / gt_semantic_seg with size
        trans = Pad(size=(1200, 2000))
        results = trans(copy.deepcopy(data_info))
        assert results['img'].shape[:2] == (2000, 1200)
        assert results['gt_semantic_seg'].shape[:2] == (2000, 1200)

        # test pad img/gt_semantic_seg with size_divisor
        trans = Pad(size_divisor=11)
        results = trans(copy.deepcopy(data_info))
        assert results['img'].shape[:2] == (1342, 803)
        assert results['gt_semantic_seg'].shape[:2] == (1342, 803)

        # test pad img/gt_semantic_seg with pad_to_square
        trans = Pad(pad_to_square=True)
        results = trans(copy.deepcopy(data_info))
        assert results['img'].shape[:2] == (1333, 1333)
        assert results['gt_semantic_seg'].shape[:2] == (1333, 1333)

        # test pad img/gt_semantic_seg with pad_to_square and size_divisor
        trans = Pad(pad_to_square=True, size_divisor=11)
        results = trans(copy.deepcopy(data_info))
        assert results['img'].shape[:2] == (1342, 1342)
        assert results['gt_semantic_seg'].shape[:2] == (1342, 1342)

        # test pad img/gt_semantic_seg with pad_to_square and size_divisor
        trans = Pad(pad_to_square=True, size_divisor=11)
        results = trans(copy.deepcopy(data_info))
        assert results['img'].shape[:2] == (1342, 1342)
        assert results['gt_semantic_seg'].shape[:2] == (1342, 1342)

        # test padding_mode
        new_img = np.ones((1333, 800, 3))
        data_info['img'] = new_img
        trans = Pad(pad_to_square=True, padding_mode='edge')
        results = trans(copy.deepcopy(data_info))
        assert (results['img'] == np.ones((1333, 1333, 3))).all()

        # test pad_val is dict
        # test rgb image, size=(2000, 2000)
        trans = Pad(
            size=(2000, 2000),
            pad_val=dict(img=(12, 12, 12), seg=(10, 10, 10)))
        results = trans(copy.deepcopy(data_info))
        assert (results['img'][1333:2000, 800:2000, :] == 12).all()
        assert (results['gt_semantic_seg'][1333:2000, 800:2000, :] == 10).all()

        trans = Pad(size=(2000, 2000), pad_val=dict(img=(12, 12, 12)))
        results = trans(copy.deepcopy(data_info))
        assert (results['img'][1333:2000, 800:2000, :] == 12).all()
        assert (results['gt_semantic_seg'][1333:2000,
                                           800:2000, :] == 255).all()

        # test rgb image, pad_to_square=True
        trans = Pad(
            pad_to_square=True,
            pad_val=dict(img=(12, 12, 12), seg=(10, 10, 10)))
        results = trans(copy.deepcopy(data_info))
        assert (results['img'][:, 800:1333, :] == 12).all()
        assert (results['gt_semantic_seg'][:, 800:1333, :] == 10).all()

        trans = Pad(pad_to_square=True, pad_val=dict(img=(12, 12, 12)))
        results = trans(copy.deepcopy(data_info))
        assert (results['img'][:, 800:1333, :] == 12).all()
        assert (results['gt_semantic_seg'][:, 800:1333, :] == 255).all()

        # test pad_val is int
        # test rgb image
        trans = Pad(size=(2000, 2000), pad_val=12)
        results = trans(copy.deepcopy(data_info))
        assert (results['img'][1333:2000, 800:2000, :] == 12).all()
        assert (results['gt_semantic_seg'][1333:2000,
                                           800:2000, :] == 255).all()
        # test gray image
        new_img = np.random.random((1333, 800))
        data_info['img'] = new_img
        new_semantic_seg = np.random.random((1333, 800))
        data_info['gt_semantic_seg'] = new_semantic_seg
        trans = Pad(size=(2000, 2000), pad_val=12)
        results = trans(copy.deepcopy(data_info))
        assert (results['img'][1333:2000, 800:2000] == 12).all()
        assert (results['gt_semantic_seg'][1333:2000, 800:2000] == 255).all()

    def test_repr(self):
        trans = Pad(pad_to_square=True, size_divisor=11, padding_mode='edge')
        assert repr(trans) == (
            'Pad(size=None, size_divisor=11, pad_to_square=True, '
            "pad_val={'img': 0, 'seg': 255}), padding_mode=edge)")


class TestCenterCrop:

    @classmethod
    def setup_class(cls):
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
        cls.original_img = copy.deepcopy(img)
        seg = np.random.randint(0, 19, (300, 400)).astype(np.uint8)
        cls.gt_semantic_map = copy.deepcopy(seg)

    @staticmethod
    def reset_results(results, original_img, gt_semantic_map):
        results['img'] = copy.deepcopy(original_img)
        results['gt_semantic_seg'] = copy.deepcopy(gt_semantic_map)
        results['gt_bboxes'] = np.array([[0, 0, 210, 160],
                                         [200, 150, 400, 300]])
        results['gt_keypoints'] = np.array([[[20, 50, 1]], [[200, 150, 1]],
                                            [[300, 225, 1]]])
        return results

    @pytest.mark.skipif(
        condition=torch is None, reason='No torch in current env')
    def test_error(self):
        # test assertion if size is smaller than 0
        with pytest.raises(AssertionError):
            transform = dict(type='CenterCrop', crop_size=-1)
            TRANSFORMS.build(transform)

        # test assertion if size is tuple but one value is smaller than 0
        with pytest.raises(AssertionError):
            transform = dict(type='CenterCrop', crop_size=(224, -1))
            TRANSFORMS.build(transform)

        # test assertion if size is tuple and len(size) < 2
        with pytest.raises(AssertionError):
            transform = dict(type='CenterCrop', crop_size=(224, ))
            TRANSFORMS.build(transform)

        # test assertion if size is tuple len(size) > 2
        with pytest.raises(AssertionError):
            transform = dict(type='CenterCrop', crop_size=(224, 224, 3))
            TRANSFORMS.build(transform)

    def test_repr(self):
        # test repr
        transform = dict(type='CenterCrop', crop_size=224)
        center_crop_module = TRANSFORMS.build(transform)
        assert isinstance(repr(center_crop_module), str)

    def test_transform(self):
        results = {}
        self.reset_results(results, self.original_img, self.gt_semantic_map)

        # test CenterCrop when size is int
        transform = dict(type='CenterCrop', crop_size=224)
        center_crop_module = TRANSFORMS.build(transform)
        results = center_crop_module(results)
        assert results['height'] == 224
        assert results['width'] == 224
        assert (results['img'] == self.original_img[38:262, 88:312, ...]).all()
        assert (
            results['gt_semantic_seg'] == self.gt_semantic_map[38:262,
                                                               88:312]).all()
        assert np.equal(results['gt_bboxes'],
                        np.array([[0, 0, 122, 122], [112, 112, 224,
                                                     224]])).all()
        assert np.equal(
            results['gt_keypoints'],
            np.array([[[0, 12, 0]], [[112, 112, 1]], [[212, 187, 1]]])).all()

        # test CenterCrop when size is tuple
        transform = dict(type='CenterCrop', crop_size=(224, 224))
        center_crop_module = TRANSFORMS.build(transform)
        results = self.reset_results(results, self.original_img,
                                     self.gt_semantic_map)
        results = center_crop_module(results)
        assert results['height'] == 224
        assert results['width'] == 224
        assert (results['img'] == self.original_img[38:262, 88:312, ...]).all()
        assert (
            results['gt_semantic_seg'] == self.gt_semantic_map[38:262,
                                                               88:312]).all()
        assert np.equal(results['gt_bboxes'],
                        np.array([[0, 0, 122, 122], [112, 112, 224,
                                                     224]])).all()
        assert np.equal(
            results['gt_keypoints'],
            np.array([[[0, 12, 0]], [[112, 112, 1]], [[212, 187, 1]]])).all()

        # test CenterCrop when crop_height != crop_width
        transform = dict(type='CenterCrop', crop_size=(224, 256))
        center_crop_module = TRANSFORMS.build(transform)
        results = self.reset_results(results, self.original_img,
                                     self.gt_semantic_map)
        results = center_crop_module(results)
        assert results['height'] == 256
        assert results['width'] == 224
        assert (results['img'] == self.original_img[22:278, 88:312, ...]).all()
        assert (
            results['gt_semantic_seg'] == self.gt_semantic_map[22:278,
                                                               88:312]).all()
        assert np.equal(results['gt_bboxes'],
                        np.array([[0, 0, 122, 138], [112, 128, 224,
                                                     256]])).all()
        assert np.equal(
            results['gt_keypoints'],
            np.array([[[0, 28, 0]], [[112, 128, 1]], [[212, 203, 1]]])).all()

        # test CenterCrop when crop_size is equal to img.shape
        img_height, img_width, _ = self.original_img.shape
        transform = dict(type='CenterCrop', crop_size=(img_width, img_height))
        center_crop_module = TRANSFORMS.build(transform)
        results = self.reset_results(results, self.original_img,
                                     self.gt_semantic_map)
        results = center_crop_module(results)
        assert results['height'] == 300
        assert results['width'] == 400
        assert (results['img'] == self.original_img).all()
        assert (results['gt_semantic_seg'] == self.gt_semantic_map).all()
        assert np.equal(results['gt_bboxes'],
                        np.array([[0, 0, 210, 160], [200, 150, 400,
                                                     300]])).all()
        assert np.equal(
            results['gt_keypoints'],
            np.array([[[20, 50, 1]], [[200, 150, 1]], [[300, 225, 1]]])).all()

        # test CenterCrop when crop_size is larger than img.shape
        transform = dict(
            type='CenterCrop', crop_size=(img_width * 2, img_height * 2))
        center_crop_module = TRANSFORMS.build(transform)
        results = self.reset_results(results, self.original_img,
                                     self.gt_semantic_map)
        results = center_crop_module(results)
        assert results['height'] == 300
        assert results['width'] == 400
        assert (results['img'] == self.original_img).all()
        assert (results['gt_semantic_seg'] == self.gt_semantic_map).all()
        assert np.equal(results['gt_bboxes'],
                        np.array([[0, 0, 210, 160], [200, 150, 400,
                                                     300]])).all()
        assert np.equal(
            results['gt_keypoints'],
            np.array([[[20, 50, 1]], [[200, 150, 1]], [[300, 225, 1]]])).all()

        # test with padding
        transform = dict(
            type='CenterCrop',
            crop_size=(img_width // 2, img_height * 2),
            pad_mode='constant',
            pad_val=12)
        center_crop_module = TRANSFORMS.build(transform)
        results = self.reset_results(results, self.original_img,
                                     self.gt_semantic_map)
        results = center_crop_module(results)
        assert results['height'] == 600
        assert results['width'] == 200
        assert results['img'].shape[:2] == results['gt_semantic_seg'].shape
        assert (results['img'][300:600, 100:300, ...] == 12).all()
        assert (results['gt_semantic_seg'][300:600, 100:300] == 255).all()
        assert np.equal(results['gt_bboxes'],
                        np.array([[0, 0, 110, 160], [100, 150, 200,
                                                     300]])).all()
        assert np.equal(
            results['gt_keypoints'],
            np.array([[[0, 50, 0]], [[100, 150, 1]], [[200, 225, 0]]])).all()

        transform = dict(
            type='CenterCrop',
            crop_size=(img_width // 2, img_height * 2),
            pad_mode='constant',
            pad_val=dict(img=13, seg=33))
        center_crop_module = TRANSFORMS.build(transform)
        results = self.reset_results(results, self.original_img,
                                     self.gt_semantic_map)
        results = center_crop_module(results)
        assert results['height'] == 600
        assert results['width'] == 200
        assert (results['img'][300:600, 100:300, ...] == 13).all()
        assert (results['gt_semantic_seg'][300:600, 100:300] == 33).all()
        assert np.equal(results['gt_bboxes'],
                        np.array([[0, 0, 110, 160], [100, 150, 200,
                                                     300]])).all()
        assert np.equal(
            results['gt_keypoints'],
            np.array([[[0, 50, 0]], [[100, 150, 1]], [[200, 225, 0]]])).all()

        # test CenterCrop when crop_width is smaller than img_width
        transform = dict(
            type='CenterCrop', crop_size=(img_width // 2, img_height))
        center_crop_module = TRANSFORMS.build(transform)
        results = self.reset_results(results, self.original_img,
                                     self.gt_semantic_map)
        results = center_crop_module(results)
        assert results['height'] == img_height
        assert results['width'] == img_width // 2
        assert (results['img'] == self.original_img[:, 100:300, ...]).all()
        assert (
            results['gt_semantic_seg'] == self.gt_semantic_map[:,
                                                               100:300]).all()
        assert np.equal(results['gt_bboxes'],
                        np.array([[0, 0, 110, 160], [100, 150, 200,
                                                     300]])).all()
        assert np.equal(
            results['gt_keypoints'],
            np.array([[[0, 50, 0]], [[100, 150, 1]], [[200, 225, 0]]])).all()

        # test CenterCrop when crop_height is smaller than img_height
        transform = dict(
            type='CenterCrop', crop_size=(img_width, img_height // 2))
        center_crop_module = TRANSFORMS.build(transform)
        results = self.reset_results(results, self.original_img,
                                     self.gt_semantic_map)
        results = center_crop_module(results)
        assert results['height'] == img_height // 2
        assert results['width'] == img_width
        assert (results['img'] == self.original_img[75:225, ...]).all()
        assert (results['gt_semantic_seg'] == self.gt_semantic_map[75:225,
                                                                   ...]).all()
        assert np.equal(results['gt_bboxes'],
                        np.array([[0, 0, 210, 85], [200, 75, 400,
                                                    150]])).all()
        assert np.equal(
            results['gt_keypoints'],
            np.array([[[20, 0, 0]], [[200, 75, 1]], [[300, 150, 0]]])).all()

    @pytest.mark.skipif(
        condition=torch is None, reason='No torch in current env')
    def test_torchvision_compare(self):
        # compare results with torchvision
        results = {}
        transform = dict(type='CenterCrop', crop_size=224)
        center_crop_module = TRANSFORMS.build(transform)
        results = self.reset_results(results, self.original_img,
                                     self.gt_semantic_map)
        results = center_crop_module(results)
        center_crop_module = torchvision.transforms.CenterCrop(size=224)
        pil_img = Image.fromarray(self.original_img)
        pil_seg = Image.fromarray(self.gt_semantic_map)
        cropped_img = center_crop_module(pil_img)
        cropped_img = np.array(cropped_img)
        cropped_seg = center_crop_module(pil_seg)
        cropped_seg = np.array(cropped_seg)
        assert np.equal(results['img'], cropped_img).all()
        assert np.equal(results['gt_semantic_seg'], cropped_seg).all()


class TestRandomGrayscale:

    @classmethod
    def setup_class(cls):
        cls.img = np.random.rand(10, 10, 3).astype(np.float32)

    def test_repr(self):
        # test repr
        transform = dict(
            type='RandomGrayscale',
            prob=1.,
            channel_weights=(0.299, 0.587, 0.114),
            keep_channel=True)
        random_gray_scale_module = TRANSFORMS.build(transform)
        assert isinstance(repr(random_gray_scale_module), str)

    def test_error(self):
        # test invalid argument
        transform = dict(type='RandomGrayscale', prob=2)
        with pytest.raises(AssertionError):
            TRANSFORMS.build(transform)

    def test_transform(self):
        results = dict()
        # test rgb2gray, return the grayscale image with prob = 1.
        transform = dict(
            type='RandomGrayscale',
            prob=1.,
            channel_weights=(0.299, 0.587, 0.114),
            keep_channel=True)

        random_gray_scale_module = TRANSFORMS.build(transform)
        results['img'] = copy.deepcopy(self.img)
        img = random_gray_scale_module(results)['img']
        computed_gray = (
            self.img[:, :, 0] * 0.299 + self.img[:, :, 1] * 0.587 +
            self.img[:, :, 2] * 0.114)
        for i in range(img.shape[2]):
            assert_array_almost_equal(img[:, :, i], computed_gray, decimal=4)
        assert img.shape == (10, 10, 3)

        # test rgb2gray, return the original image with p=0.
        transform = dict(type='RandomGrayscale', prob=0.)
        random_gray_scale_module = TRANSFORMS.build(transform)
        results['img'] = copy.deepcopy(self.img)
        img = random_gray_scale_module(results)['img']
        assert_array_equal(img, self.img)
        assert img.shape == (10, 10, 3)

        # test image with one channel
        transform = dict(type='RandomGrayscale', prob=1.)
        results['img'] = self.img[:, :, 0:1]
        random_gray_scale_module = TRANSFORMS.build(transform)
        img = random_gray_scale_module(results)['img']
        assert_array_equal(img, self.img[:, :, 0:1])
        assert img.shape == (10, 10, 1)


@TRANSFORMS.register_module()
class MockFormatBundle(BaseTransform):

    def __init__(self) -> None:
        super().__init__()

    def transform(self, results):
        data_sample = Mock()
        return results['img'], data_sample


class TestMultiScaleFlipAug:

    @classmethod
    def setup_class(cls):
        cls.img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
        cls.original_img = copy.deepcopy(cls.img)

    def test_error(self):
        # test assertion if img_scale is not tuple or list of tuple
        with pytest.raises(AssertionError):
            transform = dict(
                type='MultiScaleFlipAug', img_scale=[1333, 800], transforms=[])
            TRANSFORMS.build(transform)

        # test assertion if flip_direction is not str or list of str
        with pytest.raises(AssertionError):
            transform = dict(
                type='MultiScaleFlipAug',
                img_scale=[(1333, 800)],
                flip_direction=1,
                transforms=[])
            TRANSFORMS.build(transform)

    @pytest.mark.skipif(
        condition=torch is None, reason='No torch in current env')
    def test_multi_scale_flip_aug(self):
        # test with empty transforms
        transform = dict(
            type='MultiScaleFlipAug',
            transforms=[dict(type='MockFormatBundle')],
            img_scale=[(1333, 800), (800, 600), (640, 480)],
            flip=True,
            flip_direction=['horizontal', 'vertical', 'diagonal'])
        multi_scale_flip_aug_module = TRANSFORMS.build(transform)
        results = dict()
        results['img'] = copy.deepcopy(self.original_img)
        input, data_sample = multi_scale_flip_aug_module(results)
        assert len(input) == 12
        assert len(data_sample) == 12

        # test with flip=False
        transform = dict(
            type='MultiScaleFlipAug',
            transforms=[dict(type='MockFormatBundle')],
            img_scale=[(1333, 800), (800, 600), (640, 480)],
            flip=False,
            flip_direction=['horizontal', 'vertical', 'diagonal'])
        multi_scale_flip_aug_module = TRANSFORMS.build(transform)
        results = dict()
        results['img'] = copy.deepcopy(self.original_img)
        input, data_sample = multi_scale_flip_aug_module(results)
        assert len(input) == 3
        assert len(data_sample) == 3

        # test with transforms
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True)
        transforms_cfg = [
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='MockFormatBundle')
        ]
        transform = dict(
            type='MultiScaleFlipAug',
            transforms=transforms_cfg,
            img_scale=[(1333, 800), (800, 600), (640, 480)],
            flip=True,
            flip_direction=['horizontal', 'vertical', 'diagonal'])
        multi_scale_flip_aug_module = TRANSFORMS.build(transform)
        results = dict()
        results['img'] = copy.deepcopy(self.original_img)
        input, data_sample = multi_scale_flip_aug_module(results)
        assert len(input) == 12
        assert len(data_sample) == 12

        # test with scale_factor
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True)
        transforms_cfg = [
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='MockFormatBundle')
        ]
        transform = dict(
            type='MultiScaleFlipAug',
            transforms=transforms_cfg,
            scale_factor=[0.5, 1., 2.],
            flip=True,
            flip_direction=['horizontal', 'vertical', 'diagonal'])
        multi_scale_flip_aug_module = TRANSFORMS.build(transform)
        results = dict()
        results['img'] = copy.deepcopy(self.original_img)
        input, data_sample = multi_scale_flip_aug_module(results)
        assert len(input) == 12
        assert len(data_sample) == 12

        # test no resize
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True)
        transforms_cfg = [
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='MockFormatBundle')
        ]
        transform = dict(
            type='MultiScaleFlipAug',
            transforms=transforms_cfg,
            flip=True,
            flip_direction=['horizontal', 'vertical', 'diagonal'])
        multi_scale_flip_aug_module = TRANSFORMS.build(transform)
        results = dict()
        results['img'] = copy.deepcopy(self.original_img)
        input, data_sample = multi_scale_flip_aug_module(results)
        assert len(input) == 4
        assert len(data_sample) == 4


class TestRandomMultiscaleResize:

    @classmethod
    def setup_class(cls):
        cls.img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
        cls.original_img = copy.deepcopy(cls.img)

    def reset_results(self, results):
        results['img'] = copy.deepcopy(self.original_img)
        results['gt_semantic_seg'] = copy.deepcopy(self.original_img)

    def test_repr(self):
        # test repr
        transform = dict(
            type='RandomMultiscaleResize', scales=[(1333, 800), (1333, 600)])
        random_multiscale_resize = TRANSFORMS.build(transform)
        assert isinstance(repr(random_multiscale_resize), str)

    def test_error(self):
        # test assertion if size is smaller than 0
        with pytest.raises(AssertionError):
            transform = dict(type='RandomMultiscaleResize', scales=[0.5, 1, 2])
            TRANSFORMS.build(transform)

    def test_random_multiscale_resize(self):
        results = dict()
        # test with one scale
        transform = dict(type='RandomMultiscaleResize', scales=[(1333, 800)])
        random_multiscale_resize = TRANSFORMS.build(transform)
        self.reset_results(results)
        results = random_multiscale_resize(results)
        assert results['img'].shape == (800, 1333, 3)

        # test with multi scales
        _scale_choice = [(1333, 800), (1333, 600)]
        transform = dict(type='RandomMultiscaleResize', scales=_scale_choice)
        random_multiscale_resize = TRANSFORMS.build(transform)
        self.reset_results(results)
        results = random_multiscale_resize(results)
        assert (results['img'].shape[1],
                results['img'].shape[0]) in _scale_choice

        # test keep_ratio
        transform = dict(
            type='RandomMultiscaleResize',
            scales=[(900, 600)],
            keep_ratio=True)
        random_multiscale_resize = TRANSFORMS.build(transform)
        self.reset_results(results)
        _input_ratio = results['img'].shape[0] / results['img'].shape[1]
        results = random_multiscale_resize(results)
        _output_ratio = results['img'].shape[0] / results['img'].shape[1]
        assert_array_almost_equal(_input_ratio, _output_ratio)

        # test clip_object_border
        gt_bboxes = [[200, 150, 600, 450]]
        transform = dict(
            type='RandomMultiscaleResize',
            scales=[(200, 150)],
            clip_object_border=True)
        random_multiscale_resize = TRANSFORMS.build(transform)
        self.reset_results(results)
        results['gt_bboxes'] = np.array(gt_bboxes)
        results = random_multiscale_resize(results)
        assert results['img'].shape == (150, 200, 3)
        assert np.equal(results['gt_bboxes'], np.array([[100, 75, 200,
                                                         150]])).all()

        transform = dict(
            type='RandomMultiscaleResize',
            scales=[(200, 150)],
            clip_object_border=False)
        random_multiscale_resize = TRANSFORMS.build(transform)
        self.reset_results(results)
        results['gt_bboxes'] = np.array(gt_bboxes)
        results = random_multiscale_resize(results)
        assert results['img'].shape == (150, 200, 3)
        assert np.equal(results['gt_bboxes'], np.array([[100, 75, 300,
                                                         225]])).all()


class TestRandomFlip:

    def test_init(self):

        # prob is float
        TRANSFORMS = RandomFlip(0.1)
        assert TRANSFORMS.prob == 0.1

        # prob is None
        with pytest.raises(ValueError):
            TRANSFORMS = RandomFlip(None)
            assert TRANSFORMS.prob is None

        # prob is a list
        TRANSFORMS = RandomFlip([0.1, 0.2], ['horizontal', 'vertical'])
        assert len(TRANSFORMS.prob) == 2
        assert len(TRANSFORMS.direction) == 2

        # direction is an invalid type
        with pytest.raises(ValueError):
            TRANSFORMS = RandomFlip(0.1, 1)

        # prob is an invalid type
        with pytest.raises(ValueError):
            TRANSFORMS = RandomFlip('0.1')

    def test_transform(self):

        results = {
            'img': np.random.random((224, 224, 3)),
            'gt_bboxes': np.array([[0, 1, 100, 101]]),
            'gt_keypoints': np.array([[[100, 100, 1.0]]]),
            'gt_semantic_seg': np.random.random((224, 224, 3))
        }

        # horizontal flip
        TRANSFORMS = RandomFlip([1.0], ['horizontal'])
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert (results_update['gt_bboxes'] == np.array([[124, 1, 224,
                                                          101]])).all()

        # diagnal flip
        TRANSFORMS = RandomFlip([1.0], ['diagonal'])
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert (results_update['gt_bboxes'] == np.array([[124, 123, 224,
                                                          223]])).all()

        # vertical flip
        TRANSFORMS = RandomFlip([1.0], ['vertical'])
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert (results_update['gt_bboxes'] == np.array([[0, 123, 100,
                                                          223]])).all()

        # horizontal flip when direction is None
        TRANSFORMS = RandomFlip(1.0)
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert (results_update['gt_bboxes'] == np.array([[124, 1, 224,
                                                          101]])).all()

        TRANSFORMS = RandomFlip(0.0)
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert (results_update['gt_bboxes'] == np.array([[0, 1, 100,
                                                          101]])).all()

        # flip direction is invalid in bbox flip
        with pytest.raises(ValueError):
            TRANSFORMS = RandomFlip(1.0)
            results_update = TRANSFORMS.flip_bbox(results['gt_bboxes'],
                                                  (224, 224), 'invalid')

        # flip direction is invalid in keypoints flip
        with pytest.raises(ValueError):
            TRANSFORMS = RandomFlip(1.0)
            results_update = TRANSFORMS.flip_keypoints(results['gt_keypoints'],
                                                       (224, 224), 'invalid')

    def test_repr(self):
        TRANSFORMS = RandomFlip(0.1)
        TRANSFORMS_str = str(TRANSFORMS)
        assert isinstance(TRANSFORMS_str, str)


class TestRandomResize:

    def test_init(self):
        TRANSFORMS = RandomResize(
            (224, 224),
            (1.0, 2.0),
        )
        assert TRANSFORMS.scale == (224, 224)

    def test_repr(self):
        TRANSFORMS = RandomResize(
            (224, 224),
            (1.0, 2.0),
        )
        TRANSFORMS_str = str(TRANSFORMS)
        assert isinstance(TRANSFORMS_str, str)

    def test_transform(self):

        # choose target scale from init when override is True
        results = {}
        TRANSFORMS = RandomResize((224, 224), (1.0, 2.0))
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert results_update['scale'][0] >= 224 and results_update['scale'][
            0] <= 448
        assert results_update['scale'][1] >= 224 and results_update['scale'][
            1] <= 448

        # keep ratio is True
        results = {
            'img': np.random.random((224, 224, 3)),
            'gt_semantic_seg': np.random.random((224, 224, 3)),
            'gt_bboxes': np.array([[0, 0, 112, 112]]),
            'gt_keypoints': np.array([[[112, 112]]])
        }

        TRANSFORMS = RandomResize(
            (224, 224), (1.0, 2.0),
            resize_cfg=dict(type='Resize', keep_ratio=True))
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert 224 <= results_update['height']
        assert 448 >= results_update['height']
        assert 224 <= results_update['width']
        assert 448 >= results_update['width']
        assert results_update['keep_ratio']
        assert results['gt_bboxes'][0][2] >= 112
        assert results['gt_bboxes'][0][2] <= 112

        # keep ratio is False
        TRANSFORMS = RandomResize(
            (224, 224), (1.0, 2.0),
            resize_cfg=dict(type='Resize', keep_ratio=False))
        results_update = TRANSFORMS.transform(copy.deepcopy(results))

        # choose target scale from init when override is False and scale is a
        # list of tuples
        results = {}
        TRANSFORMS = RandomResize([(224, 448), (112, 224)],
                                  resize_cfg=dict(
                                      type='Resize', keep_ratio=True))
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert results_update['scale'][0] >= 224 and results_update['scale'][
            0] <= 448
        assert results_update['scale'][1] >= 112 and results_update['scale'][
            1] <= 224

        # the type of scale is invalid in init
        with pytest.raises(NotImplementedError):
            results = {}
            TRANSFORMS = RandomResize([(224, 448), [112, 224]],
                                      resize_cfg=dict(
                                          type='Resize', keep_ratio=True))
            results_update = TRANSFORMS.transform(copy.deepcopy(results))
