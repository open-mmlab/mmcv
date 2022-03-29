# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import numpy as np

from mmcv.transforms import LoadAnnotation, LoadImageFromFile


class TestLoadImageFromFile:

    def test_load_img(self):
        data_prefix = osp.join(osp.dirname(__file__), '../data')

        results = dict(img_path=osp.join(data_prefix, 'color.jpg'))
        transform = LoadImageFromFile()
        results = transform(copy.deepcopy(results))
        assert results['img_path'] == osp.join(data_prefix, 'color.jpg')
        assert results['img'].shape == (300, 400, 3)
        assert results['img'].dtype == np.uint8
        assert results['height'] == 300
        assert results['width'] == 400
        assert results['ori_height'] == 300
        assert results['ori_width'] == 400
        assert repr(transform) == transform.__class__.__name__ + \
            "(to_float32=False, color_type='color', " + \
            "imdecode_backend='cv2', file_client_args={'backend': 'disk'})"

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


class TestLoadAnnotation:

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

    def test_load_bboxes(self):
        transform = LoadAnnotation(
            with_bbox=True,
            with_label=False,
            with_seg=False,
            with_keypoints=False,
        )
        results = transform(copy.deepcopy(self.results))
        assert 'gt_bboxes' in results
        assert (results['gt_bboxes'] == np.array([[0, 0, 10, 20],
                                                  [10, 10, 110, 120]])).all()

    def test_load_labels(self):
        transform = LoadAnnotation(
            with_bbox=False,
            with_label=True,
            with_seg=False,
            with_keypoints=False,
        )
        results = transform(copy.deepcopy(self.results))
        assert 'gt_bboxes_labels' in results
        assert (results['gt_bboxes_labels'] == np.array([1, 2])).all()

    def test_load_kps(self):
        transform = LoadAnnotation(
            with_bbox=False,
            with_label=False,
            with_seg=False,
            with_keypoints=True,
        )
        results = transform(copy.deepcopy(self.results))
        assert 'gt_keypoints' in results
        assert (results['gt_keypoints'] == np.array([[[1, 2, 3]],
                                                     [[4, 5, 6]]])).all()

    def test_load_seg_map(self):
        transform = LoadAnnotation(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
        )
        results = transform(copy.deepcopy(self.results))
        assert 'gt_seg_map' in results
        assert results['gt_seg_map'].shape[:2] == (300, 400)

    def test_repr(self):
        transform = LoadAnnotation(
            with_bbox=True,
            with_label=False,
            with_seg=False,
            with_keypoints=False,
        )
        assert repr(transform) == (
            'LoadAnnotation(with_bbox=True, '
            'with_label=False, with_seg=False, '
            "with_keypoints=False, imdecode_backend='cv2', "
            "file_client_args={'backend': 'disk'})")
