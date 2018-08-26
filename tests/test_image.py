import os
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal


class TestImage(object):

    @classmethod
    def setup_class(cls):
        # the test img resolution is 400x300
        cls.img_path = osp.join(osp.dirname(__file__), 'data/color.jpg')
        cls.gray_img_path = osp.join(
            osp.dirname(__file__), 'data/grayscale.jpg')

    def assert_img_equal(self, img, ref_img, ratio_thr=0.999):
        assert img.shape == ref_img.shape
        assert img.dtype == ref_img.dtype
        area = ref_img.shape[0] * ref_img.shape[1]
        diff = np.abs(img.astype('int32') - ref_img.astype('int32'))
        assert np.sum(diff <= 1) / float(area) > ratio_thr

    def test_read_img(self):
        img = mmcv.read_img(self.img_path)
        assert img.shape == (300, 400, 3)
        img = mmcv.read_img(self.img_path, 'grayscale')
        assert img.shape == (300, 400)
        img = mmcv.read_img(self.gray_img_path)
        assert img.shape == (300, 400, 3)
        img = mmcv.read_img(self.gray_img_path, 'unchanged')
        assert img.shape == (300, 400)
        img = mmcv.read_img(img)
        assert_array_equal(img, mmcv.read_img(img))
        with pytest.raises(TypeError):
            mmcv.read_img(1)

    def test_img_from_bytes(self):
        with open(self.img_path, 'rb') as f:
            img_bytes = f.read()
        img = mmcv.img_from_bytes(img_bytes)
        assert img.shape == (300, 400, 3)

    def test_write_img(self):
        img = mmcv.read_img(self.img_path)
        out_file = osp.join(tempfile.gettempdir(), 'mmcv_test.jpg')
        mmcv.write_img(img, out_file)
        rewrite_img = mmcv.read_img(out_file)
        os.remove(out_file)
        self.assert_img_equal(img, rewrite_img)

    def test_bgr2gray(self):
        in_img = np.random.rand(10, 10, 3).astype(np.float32)
        out_img = mmcv.bgr2gray(in_img)
        computed_gray = (in_img[:, :, 0] * 0.114 + in_img[:, :, 1] * 0.587 +
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

    def test_scale_size(self):
        assert mmcv.scale_size((300, 200), 0.5) == (150, 100)
        assert mmcv.scale_size((11, 22), 0.7) == (8, 15)

    def test_resize(self):
        resized_img = mmcv.resize(self.img_path, (1000, 600))
        assert resized_img.shape == (600, 1000, 3)
        resized_img, w_scale, h_scale = mmcv.resize(self.img_path, (1000, 600),
                                                    True)
        assert (resized_img.shape == (600, 1000, 3) and w_scale == 2.5
                and h_scale == 2.0)
        for mode in ['nearest', 'bilinear', 'bicubic', 'area', 'lanczos']:
            resized_img = mmcv.resize(
                self.img_path, (1000, 600), interpolation=mode)
            assert resized_img.shape == (600, 1000, 3)

    def test_resize_like(self):
        a = np.zeros((100, 200, 3))
        resized_img = mmcv.resize_like(self.img_path, a)
        assert resized_img.shape == (100, 200, 3)

    def test_resize_by_ratio(self):
        resized_img = mmcv.resize_by_ratio(self.img_path, 1.5)
        assert resized_img.shape == (450, 600, 3)
        resized_img = mmcv.resize_by_ratio(self.img_path, 0.934)
        assert resized_img.shape == (280, 374, 3)

    def test_resize_keep_ar(self):
        # resize (400, 300) to (max_1000, max_600)
        resized_img = mmcv.resize_keep_ar(self.img_path, 1000, 600)
        assert resized_img.shape == (600, 800, 3)
        resized_img, scale = mmcv.resize_keep_ar(self.img_path, 1000, 600,
                                                 True)
        assert resized_img.shape == (600, 800, 3) and scale == 2.0
        # resize (400, 300) to (max_200, max_180)
        img = mmcv.read_img(self.img_path)
        resized_img = mmcv.resize_keep_ar(img, 200, 180)
        assert resized_img.shape == (150, 200, 3)
        resized_img, scale = mmcv.resize_keep_ar(self.img_path, 200, 180, True)
        assert resized_img.shape == (150, 200, 3) and scale == 0.5
        # max_long_edge cannot be less than max_short_edge
        with pytest.raises(ValueError):
            mmcv.resize_keep_ar(self.img_path, 500, 600)

    def test_limit_size(self):
        # limit to 800
        resized_img = mmcv.limit_size(self.img_path, 800)
        assert resized_img.shape == (300, 400, 3)
        resized_img, scale = mmcv.limit_size(self.img_path, 800, True)
        assert resized_img.shape == (300, 400, 3) and scale == 1
        # limit to 200
        resized_img = mmcv.limit_size(self.img_path, 200)
        assert resized_img.shape == (150, 200, 3)
        resized_img, scale = mmcv.limit_size(self.img_path, 200, True)
        assert resized_img.shape == (150, 200, 3) and scale == 0.5
        # test with img rather than img path
        img = mmcv.read_img(self.img_path)
        resized_img = mmcv.limit_size(img, 200)
        assert resized_img.shape == (150, 200, 3)
        resized_img, scale = mmcv.limit_size(img, 200, True)
        assert resized_img.shape == (150, 200, 3) and scale == 0.5

    def test_crop_img(self):
        img = mmcv.read_img(self.img_path)
        # yapf: disable
        bboxes = np.array([[100, 100, 199, 199],  # center
                           [0, 0, 150, 100],  # left-top corner
                           [250, 200, 399, 299],  # right-bottom corner
                           [0, 100, 399, 199],  # wide
                           [150, 0, 299, 299]])  # tall
        # yapf: enable
        # crop one bbox
        patch = mmcv.crop_img(img, bboxes[0, :])
        patches = mmcv.crop_img(img, bboxes[[0], :])
        assert patch.shape == (100, 100, 3)
        patch_path = osp.join(osp.dirname(__file__), 'data/patches')
        ref_patch = np.load(patch_path + '/0.npy')
        self.assert_img_equal(patch, ref_patch)
        assert isinstance(patches, list) and len(patches) == 1
        self.assert_img_equal(patches[0], ref_patch)
        # crop with no scaling and padding
        patches = mmcv.crop_img(img, bboxes)
        assert len(patches) == bboxes.shape[0]
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + '/{}.npy'.format(i))
            self.assert_img_equal(patches[i], ref_patch)
        # crop with scaling and no padding
        patches = mmcv.crop_img(img, bboxes, 1.2)
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + '/scale_{}.npy'.format(i))
            self.assert_img_equal(patches[i], ref_patch)
        # crop with scaling and padding
        patches = mmcv.crop_img(img, bboxes, 1.2, pad_fill=[255, 255, 0])
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + '/pad_{}.npy'.format(i))
            self.assert_img_equal(patches[i], ref_patch)
        patches = mmcv.crop_img(img, bboxes, 1.2, pad_fill=0)
        for i in range(len(patches)):
            ref_patch = np.load(patch_path + '/pad0_{}.npy'.format(i))
            self.assert_img_equal(patches[i], ref_patch)

    def test_pad_img(self):
        img = np.random.rand(10, 10, 3).astype(np.float32)
        padded_img = mmcv.pad_img(img, (15, 12), 0)
        assert_array_equal(img, padded_img[:10, :10, :])
        assert_array_equal(
            np.zeros((5, 12, 3), dtype='float32'), padded_img[10:, :, :])
        assert_array_equal(
            np.zeros((15, 2, 3), dtype='float32'), padded_img[:, 10:, :])
        img = np.random.randint(256, size=(10, 10, 3)).astype('uint8')
        padded_img = mmcv.pad_img(img, (15, 12, 3), [100, 110, 120])
        assert_array_equal(img, padded_img[:10, :10, :])
        assert_array_equal(
            np.array([100, 110, 120], dtype='uint8') * np.ones(
                (5, 12, 3), dtype='uint8'), padded_img[10:, :, :])
        assert_array_equal(
            np.array([100, 110, 120], dtype='uint8') * np.ones(
                (15, 2, 3), dtype='uint8'), padded_img[:, 10:, :])
        with pytest.raises(AssertionError):
            mmcv.pad_img(img, (15, ), 0)
        with pytest.raises(AssertionError):
            mmcv.pad_img(img, (5, 5), 0)
        with pytest.raises(AssertionError):
            mmcv.pad_img(img, (5, 5), [0, 1])

    def test_rotate_img(self):
        img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.uint8)
        assert_array_equal(mmcv.rotate_img(img, 0), img)
        img_r = np.array([[7, 4, 1], [8, 5, 2], [9, 6, 3]])
        assert_array_equal(mmcv.rotate_img(img, 90), img_r)
        img_r = np.array([[3, 6, 9], [2, 5, 8], [1, 4, 7]])
        assert_array_equal(mmcv.rotate_img(img, -90), img_r)

        img = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(np.uint8)
        img_r = np.array([[0, 6, 2, 0], [0, 7, 3, 0]])
        assert_array_equal(mmcv.rotate_img(img, 90), img_r)
        img_r = np.array([[1, 0, 0, 0], [2, 0, 0, 0]])
        assert_array_equal(mmcv.rotate_img(img, 90, center=(0, 0)), img_r)
        img_r = np.array([[255, 6, 2, 255], [255, 7, 3, 255]])
        assert_array_equal(mmcv.rotate_img(img, 90, border_value=255), img_r)
        img_r = np.array([[5, 1], [6, 2], [7, 3], [8, 4]])
        assert_array_equal(mmcv.rotate_img(img, 90, auto_bound=True), img_r)
