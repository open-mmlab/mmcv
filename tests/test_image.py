import os
import os.path as osp
import tempfile

import cv2
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

import mmcv


class TestImage(object):

    @classmethod
    def setup_class(cls):
        # the test img resolution is 400x300
        cls.img_path = osp.join(osp.dirname(__file__), 'data/color.jpg')
        cls.gray_img_path = osp.join(
            osp.dirname(__file__), 'data/grayscale.jpg')
        cls.img = cv2.imread(cls.img_path)

    def assert_img_equal(self, img, ref_img, ratio_thr=0.999):
        assert img.shape == ref_img.shape
        assert img.dtype == ref_img.dtype
        area = ref_img.shape[0] * ref_img.shape[1]
        diff = np.abs(img.astype('int32') - ref_img.astype('int32'))
        assert np.sum(diff <= 1) / float(area) > ratio_thr

    def test_imread(self):
        img = mmcv.imread(self.img_path)
        assert img.shape == (300, 400, 3)
        img = mmcv.imread(self.img_path, 'grayscale')
        assert img.shape == (300, 400)
        img = mmcv.imread(self.gray_img_path)
        assert img.shape == (300, 400, 3)
        img = mmcv.imread(self.gray_img_path, 'unchanged')
        assert img.shape == (300, 400)
        img = mmcv.imread(img)
        assert_array_equal(img, mmcv.imread(img))
        with pytest.raises(TypeError):
            mmcv.imread(1)

    def test_imfrombytes(self):
        with open(self.img_path, 'rb') as f:
            img_bytes = f.read()
        img = mmcv.imfrombytes(img_bytes)
        assert img.shape == (300, 400, 3)

    def test_imwrite(self):
        img = mmcv.imread(self.img_path)
        out_file = osp.join(tempfile.gettempdir(), 'mmcv_test.jpg')
        mmcv.imwrite(img, out_file)
        rewrite_img = mmcv.imread(out_file)
        os.remove(out_file)
        self.assert_img_equal(img, rewrite_img)

    def test_bgr2gray(self):
        in_img = np.random.rand(10, 10, 3).astype(np.float32)
        out_img = mmcv.bgr2gray(in_img)
        computed_gray = (
            in_img[:, :, 0] * 0.114 + in_img[:, :, 1] * 0.587 +
            in_img[:, :, 2] * 0.299)
        assert_array_almost_equal(out_img, computed_gray, decimal=4)
        out_img_3d = mmcv.bgr2gray(in_img, True)
        assert out_img_3d.shape == (10, 10, 1)
        assert_array_almost_equal(out_img_3d[..., 0], out_img, decimal=4)

    def test_gray2bgr(self):
        in_img = np.random.rand(10, 10).astype(np.float32)
        out_img = mmcv.gray2bgr(in_img)
        assert out_img.shape == (10, 10, 3)
        for i in range(3):
            assert_array_almost_equal(out_img[..., i], in_img, decimal=4)

    def test_bgr2rgb(self):
        in_img = np.random.rand(10, 10, 3).astype(np.float32)
        out_img = mmcv.bgr2rgb(in_img)
        assert out_img.shape == in_img.shape
        assert_array_equal(out_img[..., 0], in_img[..., 2])
        assert_array_equal(out_img[..., 1], in_img[..., 1])
        assert_array_equal(out_img[..., 2], in_img[..., 0])

    def test_rgb2bgr(self):
        in_img = np.random.rand(10, 10, 3).astype(np.float32)
        out_img = mmcv.rgb2bgr(in_img)
        assert out_img.shape == in_img.shape
        assert_array_equal(out_img[..., 0], in_img[..., 2])
        assert_array_equal(out_img[..., 1], in_img[..., 1])
        assert_array_equal(out_img[..., 2], in_img[..., 0])

    def test_bgr2hsv(self):
        in_img = np.random.rand(10, 10, 3).astype(np.float32)
        out_img = mmcv.bgr2hsv(in_img)
        argmax = in_img.argmax(axis=2)
        computed_hsv = np.empty_like(in_img, dtype=in_img.dtype)
        for i in range(in_img.shape[0]):
            for j in range(in_img.shape[1]):
                b = in_img[i, j, 0]
                g = in_img[i, j, 1]
                r = in_img[i, j, 2]
                v = max(r, g, b)
                s = (v - min(r, g, b)) / v if v != 0 else 0
                if argmax[i, j] == 0:
                    h = 240 + 60 * (r - g) / (v - min(r, g, b))
                elif argmax[i, j] == 1:
                    h = 120 + 60 * (b - r) / (v - min(r, g, b))
                else:
                    h = 60 * (g - b) / (v - min(r, g, b))
                if h < 0:
                    h += 360
                computed_hsv[i, j, :] = [h, s, v]
        assert_array_almost_equal(out_img, computed_hsv, decimal=2)

    def test_bgr2hls(self):
        in_img = np.random.rand(10, 10, 3).astype(np.float32)
        out_img = mmcv.bgr2hls(in_img)
        argmax = in_img.argmax(axis=2)
        computed_hls = np.empty_like(in_img, dtype=in_img.dtype)
        for i in range(in_img.shape[0]):
            for j in range(in_img.shape[1]):
                b = in_img[i, j, 0]
                g = in_img[i, j, 1]
                r = in_img[i, j, 2]
                maxc = max(r, g, b)
                minc = min(r, g, b)
                _l = (minc + maxc) / 2.0
                if minc == maxc:
                    h = 0.0
                    s = 0.0
                if _l <= 0.5:
                    s = (maxc - minc) / (maxc + minc)
                else:
                    s = (maxc - minc) / (2.0 - maxc - minc)
                if argmax[i, j] == 2:
                    h = 60 * (g - b) / (maxc - minc)
                elif argmax[i, j] == 1:
                    h = 60 * (2.0 + (b - r) / (maxc - minc))
                else:
                    h = 60 * (4.0 + (r - g) / (maxc - minc))
                if h < 0:
                    h += 360
                computed_hls[i, j, :] = [h, _l, s]
        assert_array_almost_equal(out_img, computed_hls, decimal=2)

    def test_imresize(self):
        resized_img = mmcv.imresize(self.img, (1000, 600))
        assert resized_img.shape == (600, 1000, 3)
        resized_img, w_scale, h_scale = mmcv.imresize(self.img, (1000, 600),
                                                      True)
        assert (resized_img.shape == (600, 1000, 3) and w_scale == 2.5
                and h_scale == 2.0)
        for mode in ['nearest', 'bilinear', 'bicubic', 'area', 'lanczos']:
            resized_img = mmcv.imresize(
                self.img, (1000, 600), interpolation=mode)
            assert resized_img.shape == (600, 1000, 3)

    def test_imresize_like(self):
        a = np.zeros((100, 200, 3))
        resized_img = mmcv.imresize_like(self.img, a)
        assert resized_img.shape == (100, 200, 3)

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
        patch_path = osp.join(osp.dirname(__file__), 'data/patches')
        ref_patch = np.load(patch_path + '/0.npy')
        self.assert_img_equal(patch, ref_patch)
        assert isinstance(patches, list) and len(patches) == 1
        self.assert_img_equal(patches[0], ref_patch)

        # crop with no scaling and padding
        patches = mmcv.imcrop(self.img, bboxes)
        assert len(patches) == bboxes.shape[0]
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + '/{}.npy'.format(i))
            self.assert_img_equal(patches[i], ref_patch)

        # crop with scaling and no padding
        patches = mmcv.imcrop(self.img, bboxes, 1.2)
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + '/scale_{}.npy'.format(i))
            self.assert_img_equal(patches[i], ref_patch)

        # crop with scaling and padding
        patches = mmcv.imcrop(self.img, bboxes, 1.2, pad_fill=[255, 255, 0])
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + '/pad_{}.npy'.format(i))
            self.assert_img_equal(patches[i], ref_patch)
        patches = mmcv.imcrop(self.img, bboxes, 1.2, pad_fill=0)
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + '/pad0_{}.npy'.format(i))
            self.assert_img_equal(patches[i], ref_patch)

    def test_impad(self):
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

    def test_iminvert(self):
        img = np.array([[0, 128, 255], [1, 127, 254], [2, 129, 253]],
                       dtype=np.uint8)
        img_r = np.array([[255, 127, 0], [254, 128, 1], [253, 126, 2]],
                         dtype=np.uint8)
        assert_array_equal(mmcv.iminvert(img), img_r)
