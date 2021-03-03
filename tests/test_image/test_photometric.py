# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp

import cv2
import numpy as np
import pytest
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

    def test_adjust_color(self):
        img = np.array([[0, 128, 255], [1, 127, 254], [2, 129, 253]],
                       dtype=np.uint8)
        img = np.stack([img, img, img], axis=-1)
        assert_array_equal(mmcv.adjust_color(img), img)
        img_gray = mmcv.bgr2gray(img)
        img_r = np.stack([img_gray, img_gray, img_gray], axis=-1)
        assert_array_equal(mmcv.adjust_color(img, 0), img_r)
        assert_array_equal(mmcv.adjust_color(img, 0, 1), img_r)
        assert_array_equal(
            mmcv.adjust_color(img, 0.5, 0.5),
            np.round(np.clip((img * 0.5 + img_r * 0.5), 0,
                             255)).astype(img.dtype))
        assert_array_equal(
            mmcv.adjust_color(img, 1, 1.5),
            np.round(np.clip(img * 1 + img_r * 1.5, 0, 255)).astype(img.dtype))
        assert_array_equal(
            mmcv.adjust_color(img, 0.8, -0.6, gamma=2),
            np.round(np.clip(img * 0.8 - 0.6 * img_r + 2, 0,
                             255)).astype(img.dtype))
        assert_array_equal(
            mmcv.adjust_color(img, 0.8, -0.6, gamma=-0.6),
            np.round(np.clip(img * 0.8 - 0.6 * img_r - 0.6, 0,
                             255)).astype(img.dtype))

        # test float type of image
        img = img.astype(np.float32)
        assert_array_equal(
            np.round(mmcv.adjust_color(img, 0.8, -0.6, gamma=-0.6)),
            np.round(np.clip(img * 0.8 - 0.6 * img_r - 0.6, 0, 255)))

    def test_imequalize(self, nb_rand_test=100):

        def _imequalize(img):
            # equalize the image using PIL.ImageOps.equalize
            from PIL import ImageOps, Image
            img = Image.fromarray(img)
            equalized_img = np.asarray(ImageOps.equalize(img))
            return equalized_img

        img = np.array([[0, 128, 255], [1, 127, 254], [2, 129, 253]],
                       dtype=np.uint8)
        img = np.stack([img, img, img], axis=-1)
        equalized_img = mmcv.imequalize(img)
        assert_array_equal(equalized_img, _imequalize(img))

        # test equalize with case step=0
        img = np.array([[0, 0, 0], [120, 120, 120], [255, 255, 255]],
                       dtype=np.uint8)
        img = np.stack([img, img, img], axis=-1)
        assert_array_equal(mmcv.imequalize(img), img)

        # test equalize with randomly sampled image.
        for _ in range(nb_rand_test):
            img = np.clip(
                np.random.normal(0, 1, (1000, 1200, 3)) * 260, 0,
                255).astype(np.uint8)
            equalized_img = mmcv.imequalize(img)
            assert_array_equal(equalized_img, _imequalize(img))

    def test_adjust_brightness(self, nb_rand_test=100):

        def _adjust_brightness(img, factor):
            # adjust the brightness of image using
            # PIL.ImageEnhance.Brightness
            from PIL.ImageEnhance import Brightness
            from PIL import Image
            img = Image.fromarray(img)
            brightened_img = Brightness(img).enhance(factor)
            return np.asarray(brightened_img)

        img = np.array([[0, 128, 255], [1, 127, 254], [2, 129, 253]],
                       dtype=np.uint8)
        img = np.stack([img, img, img], axis=-1)
        # test case with factor 1.0
        assert_array_equal(mmcv.adjust_brightness(img, 1.), img)
        # test case with factor 0.0
        assert_array_equal(mmcv.adjust_brightness(img, 0.), np.zeros_like(img))
        # test adjust_brightness with randomly sampled images and factors.
        for _ in range(nb_rand_test):
            img = np.clip(
                np.random.uniform(0, 1, (1000, 1200, 3)) * 260, 0,
                255).astype(np.uint8)
            factor = np.random.uniform() + np.random.choice([0, 1])
            np.testing.assert_allclose(
                mmcv.adjust_brightness(img, factor).astype(np.int32),
                _adjust_brightness(img, factor).astype(np.int32),
                rtol=0,
                atol=1)

    def test_adjust_contrast(self, nb_rand_test=100):

        def _adjust_contrast(img, factor):
            from PIL.ImageEnhance import Contrast
            from PIL import Image
            # Image.fromarray defaultly supports RGB, not BGR.
            # convert from BGR to RGB
            img = Image.fromarray(img[..., ::-1], mode='RGB')
            contrasted_img = Contrast(img).enhance(factor)
            # convert from RGB to BGR
            return np.asarray(contrasted_img)[..., ::-1]

        img = np.array([[0, 128, 255], [1, 127, 254], [2, 129, 253]],
                       dtype=np.uint8)
        img = np.stack([img, img, img], axis=-1)
        # test case with factor 1.0
        assert_array_equal(mmcv.adjust_contrast(img, 1.), img)
        # test case with factor 0.0
        assert_array_equal(
            mmcv.adjust_contrast(img, 0.), _adjust_contrast(img, 0.))
        # test adjust_contrast with randomly sampled images and factors.
        for _ in range(nb_rand_test):
            img = np.clip(
                np.random.uniform(0, 1, (1200, 1000, 3)) * 260, 0,
                255).astype(np.uint8)
            factor = np.random.uniform() + np.random.choice([0, 1])
            # Note the gap (less_equal 1) between PIL.ImageEnhance.Contrast
            # and mmcv.adjust_contrast comes from the gap that converts from
            # a color image to gray image using mmcv or PIL.
            np.testing.assert_allclose(
                mmcv.adjust_contrast(img, factor).astype(np.int32),
                _adjust_contrast(img, factor).astype(np.int32),
                rtol=0,
                atol=1)

    def test_lut_transform(self):
        lut_table = np.array(list(range(256)))

        # test assertion image values should between 0 and 255.
        with pytest.raises(AssertionError):
            mmcv.lut_transform(np.array([256]), lut_table)
        with pytest.raises(AssertionError):
            mmcv.lut_transform(np.array([-1]), lut_table)

        # test assertion lut_table should be ndarray with shape (256, )
        with pytest.raises(AssertionError):
            mmcv.lut_transform(np.array([0]), list(range(256)))
        with pytest.raises(AssertionError):
            mmcv.lut_transform(np.array([1]), np.array(list(range(257))))

        img = mmcv.lut_transform(self.img, lut_table)
        baseline = cv2.LUT(self.img, lut_table)
        assert np.allclose(img, baseline)

        input_img = np.array(
            [[[0, 128, 255], [255, 128, 0]], [[0, 128, 255], [255, 128, 0]]],
            dtype=np.float)
        img = mmcv.lut_transform(input_img, lut_table)
        baseline = cv2.LUT(np.array(input_img, dtype=np.uint8), lut_table)
        assert np.allclose(img, baseline)

        input_img = np.random.randint(0, 256, size=(7, 8, 9, 10, 11))
        img = mmcv.lut_transform(input_img, lut_table)
        baseline = cv2.LUT(np.array(input_img, dtype=np.uint8), lut_table)
        assert np.allclose(img, baseline)

    def test_clahe(self):

        def _clahe(img, clip_limit=40.0, tile_grid_size=(8, 8)):
            clahe = cv2.createCLAHE(clip_limit, tile_grid_size)
            return clahe.apply(np.array(img, dtype=np.uint8))

        # test assertion image should have the right shape
        with pytest.raises(AssertionError):
            mmcv.clahe(self.img)

        # test assertion tile_grid_size should be a tuple with 2 integers
        with pytest.raises(AssertionError):
            mmcv.clahe(self.img[:, :, 0], tile_grid_size=(8.0, 8.0))
        with pytest.raises(AssertionError):
            mmcv.clahe(self.img[:, :, 0], tile_grid_size=(8, 8, 8))
        with pytest.raises(AssertionError):
            mmcv.clahe(self.img[:, :, 0], tile_grid_size=[8, 8])

        # test with different channels
        for i in range(self.img.shape[-1]):
            img = mmcv.clahe(self.img[:, :, i])
            img_std = _clahe(self.img[:, :, i])
            assert np.allclose(img, img_std)
            assert id(img) != id(self.img[:, :, i])
            assert id(img_std) != id(self.img[:, :, i])

        # test case with clip_limit=1.2
        for i in range(self.img.shape[-1]):
            img = mmcv.clahe(self.img[:, :, i], 1.2)
            img_std = _clahe(self.img[:, :, i], 1.2)
            assert np.allclose(img, img_std)
            assert id(img) != id(self.img[:, :, i])
            assert id(img_std) != id(self.img[:, :, i])
