# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import numpy as np
import pytest

import mmcv
from mmcv.transforms import Normalize, Pad, RandomFlip, RandomResize, Resize


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

        # test pad_val
        new_img = np.zeros((1333, 800, 3))
        data_info['img'] = new_img
        trans = Pad(pad_to_square=True, pad_val=0)
        results = trans(copy.deepcopy(data_info))
        assert (results['img'] == np.zeros((1333, 1333, 3))).all()

    def test_repr(self):
        trans = Pad(pad_to_square=True, size_divisor=11, padding_mode='edge')
        assert repr(trans) == (
            'Pad(size=None, size_divisor=11, pad_to_square=True, '
            "pad_val={'img': 0, 'seg': 255}), padding_mode=edge)")


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
            results_update = TRANSFORMS.bbox_flip(results['gt_bboxes'],
                                                  (224, 224), 'invalid')

        # flip direction is invalid in keypoints flip
        with pytest.raises(ValueError):
            TRANSFORMS = RandomFlip(1.0)
            results_update = TRANSFORMS.keypoints_flip(results['gt_keypoints'],
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
        # import pdb
        # pdb.set_trace()
        TRANSFORMS = RandomResize((224, 224), (1.0, 2.0), keep_ratio=True)
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert 224 <= results_update['height']
        assert 448 >= results_update['height']
        assert 224 <= results_update['width']
        assert 448 >= results_update['width']
        assert results_update['keep_ratio']
        assert results['gt_bboxes'][0][2] >= 112
        assert results['gt_bboxes'][0][2] <= 112

        # keep ratio is False
        TRANSFORMS = RandomResize((224, 224), (1.0, 2.0), keep_ratio=False)
        results_update = TRANSFORMS.transform(copy.deepcopy(results))

        # choose target scale from init when override is False and scale is a
        # list of tuples
        results = {}
        TRANSFORMS = RandomResize([(224, 448), (112, 224)], keep_ratio=True)
        results_update = TRANSFORMS.transform(copy.deepcopy(results))
        assert results_update['scale'][0] >= 224 and results_update['scale'][
            0] <= 448
        assert results_update['scale'][1] >= 112 and results_update['scale'][
            1] <= 224

        # the type of scale is invalid in init
        with pytest.raises(NotImplementedError):
            results = {}
            TRANSFORMS = RandomResize([(224, 448), [112, 224]],
                                      keep_ratio=True)
            results_update = TRANSFORMS.transform(copy.deepcopy(results))