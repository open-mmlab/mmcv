# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp

import cv2
import numpy as np
from numpy.testing import assert_array_equal

import mmcv


class TestPhotometric:

    @classmethod
    def setup_class(cls):
        # the test img resolution is 400x300
        cls.img_path = osp.join(osp.dirname(__file__), '../data/color.jpg')
        cls.img = cv2.imread(cls.img_path)
        cls.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        cls.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def test_imnormalize(self):
        rgb_img = self.img[:, :, ::-1]
        baseline = (rgb_img - self.mean) / self.std
        img = mmcv.imnormalize(self.img, self.mean, self.std)
        assert np.allclose(img, baseline)
        assert id(img) != id(self.img)
        img = mmcv.imnormalize(rgb_img, self.mean, self.std, to_rgb=False)
        assert np.allclose(img, baseline)
        assert id(img) != id(rgb_img)

    def test_imnormalize_(self):
        img_for_normalize = np.float32(self.img)
        rgb_img_for_normalize = np.float32(self.img[:, :, ::-1])
        baseline = (rgb_img_for_normalize - self.mean) / self.std
        img = mmcv.imnormalize_(img_for_normalize, self.mean, self.std)
        assert np.allclose(img_for_normalize, baseline)
        assert id(img) == id(img_for_normalize)
        img = mmcv.imnormalize_(
            rgb_img_for_normalize, self.mean, self.std, to_rgb=False)
        assert np.allclose(img, baseline)
        assert id(img) == id(rgb_img_for_normalize)

    def test_imdenormalize(self):
        norm_img = (self.img[:, :, ::-1] - self.mean) / self.std
        rgb_baseline = (norm_img * self.std + self.mean)
        bgr_baseline = rgb_baseline[:, :, ::-1]
        img = mmcv.imdenormalize(norm_img, self.mean, self.std)
        assert np.allclose(img, bgr_baseline)
        img = mmcv.imdenormalize(norm_img, self.mean, self.std, to_bgr=False)
        assert np.allclose(img, rgb_baseline)

    def test_iminvert(self):
        img = np.array([[0, 128, 255], [1, 127, 254], [2, 129, 253]],
                       dtype=np.uint8)
        img_r = np.array([[255, 127, 0], [254, 128, 1], [253, 126, 2]],
                         dtype=np.uint8)
        assert_array_equal(mmcv.iminvert(img), img_r)

    def test_solarize(self):
        img = np.array([[0, 128, 255], [1, 127, 254], [2, 129, 253]],
                       dtype=np.uint8)
        img_r = np.array([[0, 127, 0], [1, 127, 1], [2, 126, 2]],
                         dtype=np.uint8)
        assert_array_equal(mmcv.solarize(img), img_r)
        img_r = np.array([[0, 127, 0], [1, 128, 1], [2, 126, 2]],
                         dtype=np.uint8)
        assert_array_equal(mmcv.solarize(img, 100), img_r)

    def test_posterize(self):
        img = np.array([[0, 128, 255], [1, 127, 254], [2, 129, 253]],
                       dtype=np.uint8)
        img_r = np.array([[0, 128, 128], [0, 0, 128], [0, 128, 128]],
                         dtype=np.uint8)
        assert_array_equal(mmcv.posterize(img, 1), img_r)
        img_r = np.array([[0, 128, 224], [0, 96, 224], [0, 128, 224]],
                         dtype=np.uint8)
        assert_array_equal(mmcv.posterize(img, 3), img_r)

    def test_color(self):
        img = np.array([[0, 128, 255], [1, 127, 254], [2, 129, 253]],
                       dtype=np.uint8)
        img = np.stack([img, img, img], axis=-1)
        assert_array_equal(mmcv.color(img), img)
        img_gray = mmcv.bgr2gray(img)
        img_r = np.stack([img_gray, img_gray, img_gray], axis=-1)
        assert_array_equal(mmcv.color(img, 0), img_r)
        assert_array_equal(mmcv.color(img, 0, 1), img_r)
        assert_array_equal(
            mmcv.color(img, 0.5, 0.5),
            np.round(np.clip((img * 0.5 + img_r * 0.5), 0,
                             255)).astype(img.dtype))
        assert_array_equal(
            mmcv.color(img, 1, 1.5),
            np.round(np.clip(img * 1 + img_r * 1.5, 0, 255)).astype(img.dtype))
        assert_array_equal(
            mmcv.color(img, 0.8, -0.6, gamma=2),
            np.round(np.clip(img * 0.8 - 0.6 * img_r + 2, 0,
                             255)).astype(img.dtype))
        assert_array_equal(
            mmcv.color(img, 0.8, -0.6, gamma=-0.6),
            np.round(np.clip(img * 0.8 - 0.6 * img_r - 0.6, 0,
                             255)).astype(img.dtype))


obj = TestPhotometric()
obj.test_color()
