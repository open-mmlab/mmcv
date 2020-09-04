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

        # test pillow resize
        for mode in [
                'nearest', 'bilinear', 'bicubic', 'box', 'lanczos', 'hamming'
        ]:
            resized_img = mmcv.imresize(
                self.img, (1000, 600), interpolation=mode, backend='pillow')
            assert resized_img.shape == (600, 1000, 3)

        # resize backend must be 'cv2' or 'pillow'
        with pytest.raises(ValueError):
            mmcv.imresize(self.img, (1000, 600), backend='not support')

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
        # direction must be "horizontal" or "vertical" or "diagonal"
        with pytest.raises(AssertionError):
            mmcv.imflip(np.random.rand(80, 60, 3), direction='random')

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

        # test diagonal flip (color image)
        flipped_img = mmcv.imflip(img, direction='diagonal')
        assert flipped_img.shape == img.shape
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    assert flipped_img[i, j, k] == img[h - 1 - i, w - 1 - j, k]

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

        # test diagonal flip (grayscale image)
        flipped_img = mmcv.imflip(img, direction='diagonal')
        assert flipped_img.shape == img.shape
        for i in range(h):
            for j in range(w):
                assert flipped_img[i, j] == img[h - 1 - i, w - 1 - j]

    def test_imflip_(self):
        # direction must be "horizontal" or "vertical" or "diagonal"
        with pytest.raises(AssertionError):
            mmcv.imflip_(np.random.rand(80, 60, 3), direction='random')

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

        # test diagonal flip (color image)
        img_for_flip = img.copy()
        flipped_img = mmcv.imflip_(img_for_flip, direction='diagonal')
        assert flipped_img.shape == img.shape
        assert flipped_img.shape == img_for_flip.shape
        assert id(flipped_img) == id(img_for_flip)
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    assert flipped_img[i, j, k] == img[h - 1 - i, w - 1 - j, k]
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

        # test diagonal flip (grayscale image)
        img_for_flip = img.copy()
        flipped_img = mmcv.imflip_(img_for_flip, direction='diagonal')
        assert flipped_img.shape == img.shape
        assert flipped_img.shape == img_for_flip.shape
        assert id(flipped_img) == id(img_for_flip)
        for i in range(h):
            for j in range(w):
                assert flipped_img[i, j] == img[h - 1 - i, w - 1 - j]
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
            ref_patch = np.load(patch_path + f'/{i}.npy')
            assert_array_equal(patches[i], ref_patch)

        # crop with scaling and no padding
        patches = mmcv.imcrop(self.img, bboxes, 1.2)
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + f'/scale_{i}.npy')
            assert_array_equal(patches[i], ref_patch)

        # crop with scaling and padding
        patches = mmcv.imcrop(self.img, bboxes, 1.2, pad_fill=[255, 255, 0])
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + f'/pad_{i}.npy')
            assert_array_equal(patches[i], ref_patch)
        patches = mmcv.imcrop(self.img, bboxes, 1.2, pad_fill=0)
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + f'/pad0_{i}.npy')
            assert_array_equal(patches[i], ref_patch)

    def test_impad(self):
        # grayscale image
        img = np.random.rand(10, 10).astype(np.float32)
        padded_img = mmcv.impad(img, padding=(0, 0, 2, 5), pad_val=0)
        assert_array_equal(img, padded_img[:10, :10])
        assert_array_equal(
            np.zeros((5, 12), dtype='float32'), padded_img[10:, :])
        assert_array_equal(
            np.zeros((15, 2), dtype='float32'), padded_img[:, 10:])

        # RGB image
        img = np.random.rand(10, 10, 3).astype(np.float32)
        padded_img = mmcv.impad(img, padding=(0, 0, 2, 5), pad_val=0)
        assert_array_equal(img, padded_img[:10, :10, :])
        assert_array_equal(
            np.zeros((5, 12, 3), dtype='float32'), padded_img[10:, :, :])
        assert_array_equal(
            np.zeros((15, 2, 3), dtype='float32'), padded_img[:, 10:, :])

        # RGB image with different values for three channels.
        img = np.random.randint(256, size=(10, 10, 3)).astype('uint8')
        padded_img = mmcv.impad(
            img, padding=(0, 0, 2, 5), pad_val=(100, 110, 120))
        assert_array_equal(img, padded_img[:10, :10, :])
        assert_array_equal(
            np.array([100, 110, 120], dtype='uint8') * np.ones(
                (5, 12, 3), dtype='uint8'), padded_img[10:, :, :])
        assert_array_equal(
            np.array([100, 110, 120], dtype='uint8') * np.ones(
                (15, 2, 3), dtype='uint8'), padded_img[:, 10:, :])

        # Pad the grayscale image to shape (15, 12)
        img = np.random.rand(10, 10).astype(np.float32)
        padded_img = mmcv.impad(img, shape=(15, 12))
        assert_array_equal(img, padded_img[:10, :10])
        assert_array_equal(
            np.zeros((5, 12), dtype='float32'), padded_img[10:, :])
        assert_array_equal(
            np.zeros((15, 2), dtype='float32'), padded_img[:, 10:])

        # Pad the RGB image to shape (15, 12)
        img = np.random.rand(10, 10, 3).astype(np.float32)
        padded_img = mmcv.impad(img, shape=(15, 12))
        assert_array_equal(img, padded_img[:10, :10, :])
        assert_array_equal(
            np.zeros((5, 12, 3), dtype='float32'), padded_img[10:, :, :])
        assert_array_equal(
            np.zeros((15, 2, 3), dtype='float32'), padded_img[:, 10:, :])

        # Pad the RGB image to shape (15, 12) with different values for
        # three channels.
        img = np.random.randint(256, size=(10, 10, 3)).astype('uint8')
        padded_img = mmcv.impad(img, shape=(15, 12), pad_val=(100, 110, 120))
        assert_array_equal(img, padded_img[:10, :10, :])
        assert_array_equal(
            np.array([100, 110, 120], dtype='uint8') * np.ones(
                (5, 12, 3), dtype='uint8'), padded_img[10:, :, :])
        assert_array_equal(
            np.array([100, 110, 120], dtype='uint8') * np.ones(
                (15, 2, 3), dtype='uint8'), padded_img[:, 10:, :])

        # RGB image with padding=[5, 2]
        img = np.random.rand(10, 10, 3).astype(np.float32)
        padded_img = mmcv.impad(img, padding=(5, 2), pad_val=0)

        assert padded_img.shape == (14, 20, 3)
        assert_array_equal(img, padded_img[2:12, 5:15, :])
        assert_array_equal(
            np.zeros((2, 5, 3), dtype='float32'), padded_img[:2, :5, :])
        assert_array_equal(
            np.zeros((2, 5, 3), dtype='float32'), padded_img[12:, :5, :])
        assert_array_equal(
            np.zeros((2, 5, 3), dtype='float32'), padded_img[:2, 15:, :])
        assert_array_equal(
            np.zeros((2, 5, 3), dtype='float32'), padded_img[12:, 15:, :])

        # RGB image with type(pad_val) = tuple
        pad_val = (0, 1, 2)
        img = np.random.rand(10, 10, 3).astype(np.float32)
        padded_img = mmcv.impad(img, padding=(0, 0, 5, 2), pad_val=pad_val)

        assert padded_img.shape == (12, 15, 3)
        assert_array_equal(img, padded_img[:10, :10, :])
        assert_array_equal(pad_val[0] * np.ones((2, 15, 1), dtype='float32'),
                           padded_img[10:, :, 0:1])
        assert_array_equal(pad_val[1] * np.ones((2, 15, 1), dtype='float32'),
                           padded_img[10:, :, 1:2])
        assert_array_equal(pad_val[2] * np.ones((2, 15, 1), dtype='float32'),
                           padded_img[10:, :, 2:3])

        assert_array_equal(pad_val[0] * np.ones((12, 5, 1), dtype='float32'),
                           padded_img[:, 10:, 0:1])
        assert_array_equal(pad_val[1] * np.ones((12, 5, 1), dtype='float32'),
                           padded_img[:, 10:, 1:2])
        assert_array_equal(pad_val[2] * np.ones((12, 5, 1), dtype='float32'),
                           padded_img[:, 10:, 2:3])

        # test different padding mode with channel number = 3
        for mode in ['constant', 'edge', 'reflect', 'symmetric']:
            img = np.random.rand(10, 10, 3).astype(np.float32)
            padded_img = mmcv.impad(
                img, padding=(0, 0, 5, 2), pad_val=pad_val, padding_mode=mode)
            assert padded_img.shape == (12, 15, 3)

        # test different padding mode with channel number = 1
        for mode in ['constant', 'edge', 'reflect', 'symmetric']:
            img = np.random.rand(10, 10).astype(np.float32)
            padded_img = mmcv.impad(
                img, padding=(0, 0, 5, 2), pad_val=0, padding_mode=mode)
            assert padded_img.shape == (12, 15)

        # Padding must be a int or a 2, or 4 element tuple.
        with pytest.raises(ValueError):
            mmcv.impad(img, padding=(1, 1, 1))

        # pad_val must be a int or a tuple
        with pytest.raises(TypeError):
            mmcv.impad(img, padding=(1, 1, 1, 1), pad_val='wrong')

        # When pad_val is a tuple,
        # len(pad_val) should be equal to img.shape[-1]
        img = np.random.rand(10, 10, 3).astype(np.float32)
        with pytest.raises(AssertionError):
            mmcv.impad(img, padding=3, pad_val=(100, 200))

        with pytest.raises(AssertionError):
            mmcv.impad(img, padding=2, pad_val=0, padding_mode='unknown')

        with pytest.raises(AssertionError):
            mmcv.impad(img, shape=(12, 15), padding=(0, 0, 5, 2))

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

    def test_imshear(self):
        img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.uint8)
        assert_array_equal(mmcv.imshear(img, 0), img)
        # magnitude=1, horizontal
        img_sheared = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 7]],
                               dtype=np.uint8)
        assert_array_equal(mmcv.imshear(img, 1), img_sheared)
        # magnitude=-1, vertical
        img_sheared = np.array([[1, 5, 9], [4, 8, 0], [7, 0, 0]],
                               dtype=np.uint8)
        assert_array_equal(mmcv.imshear(img, -1, 'vertical'), img_sheared)
        # magnitude=1, vertical, borderValue=100
        borderValue = 100
        img_sheared = np.array(
            [[1, borderValue, borderValue], [4, 2, borderValue], [7, 5, 3]],
            dtype=np.uint8)
        assert_array_equal(
            mmcv.imshear(img, 1, 'vertical', borderValue), img_sheared)
        # magnitude=1, vertical, borderValue=100, img shape (h,w,3)
        img = np.stack([img, img, img], axis=-1)
        img_sheared = np.stack([img_sheared, img_sheared, img_sheared],
                               axis=-1)
        assert_array_equal(
            mmcv.imshear(img, 1, 'vertical', borderValue), img_sheared)
        # test tuple format of borderValue
        assert_array_equal(
            mmcv.imshear(img, 1, 'vertical',
                         (borderValue, borderValue, borderValue)), img_sheared)

        # test invalid length of borderValue
        with pytest.raises(AssertionError):
            mmcv.imshear(img, 0.5, 'horizontal', (borderValue, ))

        # test invalid type of borderValue
        with pytest.raises(ValueError):
            mmcv.imshear(img, 0.5, 'horizontal', [borderValue])

        # test invalid value of direction
        with pytest.raises(AssertionError):
            mmcv.imshear(img, 0.5, 'diagonal')

    def test_imtranslate(self):
        img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        assert_array_equal(mmcv.imtranslate(img, 0), img)
        # offset=1, horizontal
        img_translated = np.array([[128, 1, 2], [128, 4, 5], [128, 7, 8]],
                                  dtype=np.uint8)
        assert_array_equal(
            mmcv.imtranslate(img, 1, border_value=128), img_translated)
        # offset=-1, vertical
        img_translated = np.array([[4, 5, 6], [7, 8, 9], [0, 0, 0]],
                                  dtype=np.uint8)
        assert_array_equal(
            mmcv.imtranslate(img, -1, 'vertical'), img_translated)
        # offset=-2, horizontal
        img = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8)
        img = np.stack([img, img, img], axis=-1)
        img_translated = [[3, 4, 128, 128], [7, 8, 128, 128]]
        img_translated = np.stack(
            [img_translated, img_translated, img_translated], axis=-1)
        assert_array_equal(
            mmcv.imtranslate(img, -2, border_value=128), img_translated)
        # offset=2, vertical
        border_value = (110, 120, 130)
        img_translated = np.stack([
            np.ones((2, 4)) * border_value[0],
            np.ones((2, 4)) * border_value[1],
            np.ones((2, 4)) * border_value[2]
        ],
                                  axis=-1).astype(np.uint8)
        assert_array_equal(
            mmcv.imtranslate(img, 2, 'vertical', border_value), img_translated)
        # test invalid number elements in border_value
        with pytest.raises(AssertionError):
            mmcv.imtranslate(img, 1, border_value=(1, ))
        # test invalid type of border_value
        with pytest.raises(ValueError):
            mmcv.imtranslate(img, 1, border_value=[1, 2, 3])
        # test invalid value of direction
        with pytest.raises(AssertionError):
            mmcv.imtranslate(img, 1, 'diagonal')
