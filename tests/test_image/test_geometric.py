# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp

import cv2
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import mmcv


class TestGeometric:

    @classmethod
    def setup_class(cls):
        cls.data_dir = osp.join(osp.dirname(__file__), '../data')
        # the test img resolution is 400x300
        cls.img_path = osp.join(cls.data_dir, 'color.jpg')
        cls.img = cv2.imread(cls.img_path)

    def test_imresize(self):
        resized_img = mmcv.imresize(self.img, (1000, 600))
        assert resized_img.shape == (600, 1000, 3)
        resized_img, w_scale, h_scale = mmcv.imresize(self.img, (1000, 600),
                                                      True)
        assert (resized_img.shape == (600, 1000, 3) and w_scale == 2.5
                and h_scale == 2.0)
        resized_img_dst = np.empty((600, 1000, 3), dtype=self.img.dtype)
        resized_img = mmcv.imresize(self.img, (1000, 600), out=resized_img_dst)
        assert id(resized_img_dst) == id(resized_img)
        assert_array_equal(resized_img_dst,
                           mmcv.imresize(self.img, (1000, 600)))
        for mode in ['nearest', 'bilinear', 'bicubic', 'area', 'lanczos']:
            resized_img = mmcv.imresize(
                self.img, (1000, 600), interpolation=mode)
            assert resized_img.shape == (600, 1000, 3)

    def test_imresize_like(self):
        a = np.zeros((100, 200, 3))
        resized_img = mmcv.imresize_like(self.img, a)
        assert resized_img.shape == (100, 200, 3)

    def test_rescale_size(self):
        new_size, scale_factor = mmcv.rescale_size((400, 300), 1.5, True)
        assert new_size == (600, 450) and scale_factor == 1.5
        new_size, scale_factor = mmcv.rescale_size((400, 300), 0.934, True)
        assert new_size == (374, 280) and scale_factor == 0.934

        new_size = mmcv.rescale_size((400, 300), 1.5)
        assert new_size == (600, 450)
        new_size = mmcv.rescale_size((400, 300), 0.934)
        assert new_size == (374, 280)

        new_size, scale_factor = mmcv.rescale_size((400, 300), (1000, 600),
                                                   True)
        assert new_size == (800, 600) and scale_factor == 2.0
        new_size, scale_factor = mmcv.rescale_size((400, 300), (180, 200),
                                                   True)
        assert new_size == (200, 150) and scale_factor == 0.5

        new_size = mmcv.rescale_size((400, 300), (1000, 600))
        assert new_size == (800, 600)
        new_size = mmcv.rescale_size((400, 300), (180, 200))
        assert new_size == (200, 150)

        with pytest.raises(ValueError):
            mmcv.rescale_size((400, 300), -0.5)
        with pytest.raises(TypeError):
            mmcv.rescale_size()((400, 300), [100, 100])

    def test_imrescale(self):
        # rescale by a certain factor
        resized_img = mmcv.imrescale(self.img, 1.5)
        assert resized_img.shape == (450, 600, 3)
        resized_img = mmcv.imrescale(self.img, 0.934)
        assert resized_img.shape == (280, 374, 3)

        # rescale by a certain max_size
        # resize (400, 300) to (max_1000, max_600)
        resized_img = mmcv.imrescale(self.img, (1000, 600))
        assert resized_img.shape == (600, 800, 3)
        resized_img, scale = mmcv.imrescale(
            self.img, (1000, 600), return_scale=True)
        assert resized_img.shape == (600, 800, 3) and scale == 2.0
        # resize (400, 300) to (max_200, max_180)
        resized_img = mmcv.imrescale(self.img, (180, 200))
        assert resized_img.shape == (150, 200, 3)
        resized_img, scale = mmcv.imrescale(
            self.img, (180, 200), return_scale=True)
        assert resized_img.shape == (150, 200, 3) and scale == 0.5

        # test exceptions
        with pytest.raises(ValueError):
            mmcv.imrescale(self.img, -0.5)
        with pytest.raises(TypeError):
            mmcv.imrescale(self.img, [100, 100])

    def test_imflip(self):
        # test horizontal flip (color image)
        img = np.random.rand(80, 60, 3)
        h, w, c = img.shape
        flipped_img = mmcv.imflip(img)
        assert flipped_img.shape == img.shape
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    assert flipped_img[i, j, k] == img[i, w - 1 - j, k]
        # test vertical flip (color image)
        flipped_img = mmcv.imflip(img, direction='vertical')
        assert flipped_img.shape == img.shape
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    assert flipped_img[i, j, k] == img[h - 1 - i, j, k]
        # test horizontal flip (grayscale image)
        img = np.random.rand(80, 60)
        h, w = img.shape
        flipped_img = mmcv.imflip(img)
        assert flipped_img.shape == img.shape
        for i in range(h):
            for j in range(w):
                assert flipped_img[i, j] == img[i, w - 1 - j]
        # test vertical flip (grayscale image)
        flipped_img = mmcv.imflip(img, direction='vertical')
        assert flipped_img.shape == img.shape
        for i in range(h):
            for j in range(w):
                assert flipped_img[i, j] == img[h - 1 - i, j]

    def test_imflip_(self):
        # test horizontal flip (color image)
        img = np.random.rand(80, 60, 3)
        h, w, c = img.shape
        img_for_flip = img.copy()
        flipped_img = mmcv.imflip_(img_for_flip)
        assert flipped_img.shape == img.shape
        assert flipped_img.shape == img_for_flip.shape
        assert id(flipped_img) == id(img_for_flip)
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    assert flipped_img[i, j, k] == img[i, w - 1 - j, k]
                    assert flipped_img[i, j, k] == img_for_flip[i, j, k]

        # test vertical flip (color image)
        img_for_flip = img.copy()
        flipped_img = mmcv.imflip_(img_for_flip, direction='vertical')
        assert flipped_img.shape == img.shape
        assert flipped_img.shape == img_for_flip.shape
        assert id(flipped_img) == id(img_for_flip)
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    assert flipped_img[i, j, k] == img[h - 1 - i, j, k]
                    assert flipped_img[i, j, k] == img_for_flip[i, j, k]

        # test horizontal flip (grayscale image)
        img = np.random.rand(80, 60)
        h, w = img.shape
        img_for_flip = img.copy()
        flipped_img = mmcv.imflip_(img_for_flip)
        assert flipped_img.shape == img.shape
        assert flipped_img.shape == img_for_flip.shape
        assert id(flipped_img) == id(img_for_flip)
        for i in range(h):
            for j in range(w):
                assert flipped_img[i, j] == img[i, w - 1 - j]
                assert flipped_img[i, j] == img_for_flip[i, j]

        # test vertical flip (grayscale image)
        img_for_flip = img.copy()
        flipped_img = mmcv.imflip_(img_for_flip, direction='vertical')
        assert flipped_img.shape == img.shape
        assert flipped_img.shape == img_for_flip.shape
        assert id(flipped_img) == id(img_for_flip)
        for i in range(h):
            for j in range(w):
                assert flipped_img[i, j] == img[h - 1 - i, j]
                assert flipped_img[i, j] == img_for_flip[i, j]

    def test_imcrop(self):
        # yapf: disable
        bboxes = np.array([[100, 100, 199, 199],  # center
                           [0, 0, 150, 100],  # left-top corner
                           [250, 200, 399, 299],  # right-bottom corner
                           [0, 100, 399, 199],  # wide
                           [150, 0, 299, 299]])  # tall
        # yapf: enable

        # crop one bbox
        patch = mmcv.imcrop(self.img, bboxes[0, :])
        patches = mmcv.imcrop(self.img, bboxes[[0], :])
        assert patch.shape == (100, 100, 3)
        patch_path = osp.join(self.data_dir, 'patches')
        ref_patch = np.load(patch_path + '/0.npy')
        assert_array_equal(patch, ref_patch)
        assert isinstance(patches, list) and len(patches) == 1
        assert_array_equal(patches[0], ref_patch)

        # crop with no scaling and padding
        patches = mmcv.imcrop(self.img, bboxes)
        assert len(patches) == bboxes.shape[0]
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + '/{}.npy'.format(i))
            assert_array_equal(patches[i], ref_patch)

        # crop with scaling and no padding
        patches = mmcv.imcrop(self.img, bboxes, 1.2)
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + '/scale_{}.npy'.format(i))
            assert_array_equal(patches[i], ref_patch)

        # crop with scaling and padding
        patches = mmcv.imcrop(self.img, bboxes, 1.2, pad_fill=[255, 255, 0])
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + '/pad_{}.npy'.format(i))
            assert_array_equal(patches[i], ref_patch)
        patches = mmcv.imcrop(self.img, bboxes, 1.2, pad_fill=0)
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + '/pad0_{}.npy'.format(i))
            assert_array_equal(patches[i], ref_patch)

    def test_impad(self):
        # grayscale image
        img = np.random.rand(10, 10).astype(np.float32)
        padded_img = mmcv.impad(img, (15, 12), 0)
        assert_array_equal(img, padded_img[:10, :10])
        assert_array_equal(
            np.zeros((5, 12), dtype='float32'), padded_img[10:, :])
        assert_array_equal(
            np.zeros((15, 2), dtype='float32'), padded_img[:, 10:])

        # RGB image
        img = np.random.rand(10, 10, 3).astype(np.float32)
        padded_img = mmcv.impad(img, (15, 12), 0)
        assert_array_equal(img, padded_img[:10, :10, :])
        assert_array_equal(
            np.zeros((5, 12, 3), dtype='float32'), padded_img[10:, :, :])
        assert_array_equal(
            np.zeros((15, 2, 3), dtype='float32'), padded_img[:, 10:, :])

        img = np.random.randint(256, size=(10, 10, 3)).astype('uint8')
        padded_img = mmcv.impad(img, (15, 12, 3), [100, 110, 120])
        assert_array_equal(img, padded_img[:10, :10, :])
        assert_array_equal(
            np.array([100, 110, 120], dtype='uint8') * np.ones(
                (5, 12, 3), dtype='uint8'), padded_img[10:, :, :])
        assert_array_equal(
            np.array([100, 110, 120], dtype='uint8') * np.ones(
                (15, 2, 3), dtype='uint8'), padded_img[:, 10:, :])

        with pytest.raises(AssertionError):
            mmcv.impad(img, (15, ), 0)
        with pytest.raises(AssertionError):
            mmcv.impad(img, (5, 5), 0)
        with pytest.raises(AssertionError):
            mmcv.impad(img, (5, 5), [0, 1])

    def test_impad_to_multiple(self):
        img = np.random.rand(11, 14, 3).astype(np.float32)
        padded_img = mmcv.impad_to_multiple(img, 4)
        assert padded_img.shape == (12, 16, 3)
        img = np.random.rand(20, 12).astype(np.float32)
        padded_img = mmcv.impad_to_multiple(img, 5)
        assert padded_img.shape == (20, 15)
        img = np.random.rand(20, 12).astype(np.float32)
        padded_img = mmcv.impad_to_multiple(img, 2)
        assert padded_img.shape == (20, 12)

    def test_imrotate(self):
        img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.uint8)
        assert_array_equal(mmcv.imrotate(img, 0), img)
        img_r = np.array([[7, 4, 1], [8, 5, 2], [9, 6, 3]])
        assert_array_equal(mmcv.imrotate(img, 90), img_r)
        img_r = np.array([[3, 6, 9], [2, 5, 8], [1, 4, 7]])
        assert_array_equal(mmcv.imrotate(img, -90), img_r)

        img = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(np.uint8)
        img_r = np.array([[0, 6, 2, 0], [0, 7, 3, 0]])
        assert_array_equal(mmcv.imrotate(img, 90), img_r)
        img_r = np.array([[1, 0, 0, 0], [2, 0, 0, 0]])
        assert_array_equal(mmcv.imrotate(img, 90, center=(0, 0)), img_r)
        img_r = np.array([[255, 6, 2, 255], [255, 7, 3, 255]])
        assert_array_equal(mmcv.imrotate(img, 90, border_value=255), img_r)
        img_r = np.array([[5, 1], [6, 2], [7, 3], [8, 4]])
        assert_array_equal(mmcv.imrotate(img, 90, auto_bound=True), img_r)

        with pytest.raises(ValueError):
            mmcv.imrotate(img, 90, center=(0, 0), auto_bound=True)
