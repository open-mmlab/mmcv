import os.path as osp

import numpy as np
import pytest
from PIL import Image

from mmcv.datasets import PIPELINES


class TestLoading(object):

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.join(osp.dirname(__file__), '../../data')

    def test_load_img(self):
        # test load
        data_info = dict(img_prefix=self.data_prefix, filename='color.jpg')
        transform_cfg = dict(type='LoadImageFromFile')
        transform = PIPELINES.build(transform_cfg)
        results = transform(data_info)
        assert results['img_fields'] == ['img']
        assert results['filename'] == osp.join(self.data_prefix, 'color.jpg')
        assert results['ori_filename'] == 'color.jpg'
        assert results['img'].shape == (300, 400, 3)
        assert results['img'].dtype == np.uint8
        assert results['img_shape'] == (300, 400, 3)
        assert results['ori_shape'] == (300, 400, 3)
        assert results['height'] == 300
        assert results['width'] == 400

        # test without filename
        with pytest.raises(AssertionError):
            data_info = dict(img_prefix=self.data_prefix)
            transform(data_info)

        # test without img_prefix
        file_path = osp.join(self.data_prefix, 'color.jpg')
        data_info = dict(filename=file_path)
        transform_cfg = dict(type='LoadImageFromFile')
        transform = PIPELINES.build(transform_cfg)
        results = transform(data_info)
        assert results['filename'] == file_path
        assert results['img'].shape == (300, 400, 3)

        # test to_float32
        transform_cfg = dict(type='LoadImageFromFile', to_float32=True)
        transform = PIPELINES.build(transform_cfg)
        results = transform(data_info)
        assert results['img'].dtype == np.float32

        # test gray image
        data_info = dict(img_prefix=self.data_prefix, filename='grayscale.jpg')
        transform_cfg = dict(type='LoadImageFromFile')
        transform = PIPELINES.build(transform_cfg)
        results = transform(data_info)
        assert results['img'].shape == (300, 400, 3)
        assert results['img'].dtype == np.uint8

        # test color_type
        data_info = dict(img_prefix=self.data_prefix, filename='grayscale.jpg')
        transform_cfg = dict(type='LoadImageFromFile', color_type='unchanged')
        transform = PIPELINES.build(transform_cfg)
        results = transform(data_info)
        assert results['img'].shape == (300, 400)
        assert results['img'].dtype == np.uint8

        # test imdecode_backend
        data_info = dict(img_prefix=self.data_prefix, filename='color.jpg')
        transform_cfg = dict(
            type='LoadImageFromFile', imdecode_backend='pillow')
        transform = PIPELINES.build(transform_cfg)
        results = transform(data_info)
        pil_img = Image.open(results['filename'])
        pil_img = np.array(pil_img)[:, :, ::-1]
        assert np.allclose(results['img'], pil_img)
