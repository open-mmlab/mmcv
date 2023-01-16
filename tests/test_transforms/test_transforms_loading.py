# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import numpy as np
import pytest

from mmcv.transforms import LoadAnnotations, LoadImageFromFile


class TestLoadImageFromFile:

    def test_load_img(self):
        # file_client_args and backend_args can not be both set
        with pytest.raises(
                ValueError,
                match='"file_client_args" and "backend_args" cannot be set'):
            LoadImageFromFile(
                file_client_args={'backend': 'disk'},
                backend_args={'backend': 'disk'})
        data_prefix = osp.join(osp.dirname(__file__), '../data')

        results = dict(img_path=osp.join(data_prefix, 'color.jpg'))
        transform = LoadImageFromFile()
        results = transform(copy.deepcopy(results))
        assert results['img_path'] == osp.join(data_prefix, 'color.jpg')
        assert results['img'].shape == (300, 400, 3)
        assert results['img'].dtype == np.uint8
        assert results['img_shape'] == (300, 400)
        assert results['ori_shape'] == (300, 400)
        assert repr(transform) == transform.__class__.__name__ + \
            "(ignore_empty=False, to_float32=False, color_type='color', " + \
            "imdecode_backend='cv2', backend_args=None)"

        # to_float32
        transform = LoadImageFromFile(to_float32=True)
        results = transform(copy.deepcopy(results))
        assert results['img'].dtype == np.float32

        # gray image
        results = dict(img_path=osp.join(data_prefix, 'grayscale.jpg'))
        transform = LoadImageFromFile()
        results = transform(copy.deepcopy(results))
        assert results['img'].shape == (300, 400, 3)
        assert results['img'].dtype == np.uint8

        transform = LoadImageFromFile(color_type='unchanged')
        results = transform(copy.deepcopy(results))
        assert results['img'].shape == (300, 400)
        assert results['img'].dtype == np.uint8

        # test load empty
        fake_img_path = osp.join(data_prefix, 'fake.jpg')
        results['img_path'] = fake_img_path
        transform = LoadImageFromFile(ignore_empty=False)
        with pytest.raises(FileNotFoundError):
            transform(copy.deepcopy(results))
        transform = LoadImageFromFile(ignore_empty=True)
        assert transform(copy.deepcopy(results)) is None


class TestLoadAnnotations:

    def setup_class(cls):
        data_prefix = osp.join(osp.dirname(__file__), '../data')
        seg_map = osp.join(data_prefix, 'grayscale.jpg')
        cls.results = {
            'seg_map_path':
            seg_map,
            'instances': [{
                'bbox': [0, 0, 10, 20],
                'bbox_label': 1,
                'keypoints': [1, 2, 3]
            }, {
                'bbox': [10, 10, 110, 120],
                'bbox_label': 2,
                'keypoints': [4, 5, 6]
            }]
        }

    def test_init(self):
        # file_client_args and backend_args can not be both set
        with pytest.raises(
                ValueError,
                match='"file_client_args" and "backend_args" cannot be set'):
            LoadAnnotations(
                file_client_args={'backend': 'disk'},
                backend_args={'backend': 'disk'})

    def test_load_bboxes(self):
        transform = LoadAnnotations(
            with_bbox=True,
            with_label=False,
            with_seg=False,
            with_keypoints=False,
        )
        results = transform(copy.deepcopy(self.results))
        assert 'gt_bboxes' in results
        assert (results['gt_bboxes'] == np.array([[0, 0, 10, 20],
                                                  [10, 10, 110, 120]])).all()
        assert results['gt_bboxes'].dtype == np.float32

    def test_load_labels(self):
        transform = LoadAnnotations(
            with_bbox=False,
            with_label=True,
            with_seg=False,
            with_keypoints=False,
        )
        results = transform(copy.deepcopy(self.results))
        assert 'gt_bboxes_labels' in results
        assert (results['gt_bboxes_labels'] == np.array([1, 2])).all()
        assert results['gt_bboxes_labels'].dtype == np.int64

    def test_load_kps(self):
        transform = LoadAnnotations(
            with_bbox=False,
            with_label=False,
            with_seg=False,
            with_keypoints=True,
        )
        results = transform(copy.deepcopy(self.results))
        assert 'gt_keypoints' in results
        assert (results['gt_keypoints'] == np.array([[[1, 2, 3]],
                                                     [[4, 5, 6]]])).all()
        assert results['gt_keypoints'].dtype == np.float32

    def test_load_seg_map(self):
        transform = LoadAnnotations(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
        )
        results = transform(copy.deepcopy(self.results))
        assert 'gt_seg_map' in results
        assert results['gt_seg_map'].shape[:2] == (300, 400)
        assert results['gt_seg_map'].dtype == np.uint8

    def test_repr(self):
        transform = LoadAnnotations(
            with_bbox=True,
            with_label=False,
            with_seg=False,
            with_keypoints=False,
        )
        assert repr(transform) == (
            'LoadAnnotations(with_bbox=True, '
            'with_label=False, with_seg=False, '
            "with_keypoints=False, imdecode_backend='cv2', "
            'backend_args=None)')
