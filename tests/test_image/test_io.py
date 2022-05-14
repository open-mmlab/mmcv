# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal

import mmcv
from mmcv.fileio.file_client import HTTPBackend, PetrelBackend

if torch.__version__ == 'parrots':
    pytest.skip('not necessary in parrots test', allow_module_level=True)


class TestIO:

    @classmethod
    def setup_class(cls):
        cls.data_dir = osp.join(osp.dirname(__file__), '../data')
        # the test img resolution is 400x300
        cls.img_path = osp.join(cls.data_dir, 'color.jpg')
        cls.img_path_obj = Path(cls.img_path)
        cls.gray_img_path = osp.join(cls.data_dir, 'grayscale.jpg')
        cls.gray_img_path_obj = Path(cls.gray_img_path)
        cls.gray_img_dim3_path = osp.join(cls.data_dir, 'grayscale_dim3.jpg')
        cls.gray_alpha_img_path = osp.join(cls.data_dir, 'gray_alpha.png')
        cls.palette_img_path = osp.join(cls.data_dir, 'palette.gif')
        cls.exif_img_path = osp.join(cls.data_dir, 'color_exif.jpg')
        cls.img = cv2.imread(cls.img_path)
        cls.tiff_path = osp.join(cls.data_dir, 'uint16-5channel.tif')
        # petrel s3 path
        cls.s3_path = 's3://path/of/your/file.jpg'
        # http path
        cls.http_path = 'http://path/of/your/file.jpg'
        # add mock package
        sys.modules['petrel_client'] = MagicMock()
        sys.modules['petrel_client.client'] = MagicMock()

    @classmethod
    def teardown_class(cls):
        # clean instances avoid to influence other unittest
        mmcv.FileClient._instances = {}

    def assert_img_equal(self, img, ref_img, ratio_thr=0.999):
        assert img.shape == ref_img.shape
        assert img.dtype == ref_img.dtype
        area = ref_img.shape[0] * ref_img.shape[1]
        diff = np.abs(img.astype('int32') - ref_img.astype('int32'))
        assert np.sum(diff <= 1) / float(area) > ratio_thr

    def test_imread(self):
        # backend cv2
        mmcv.use_backend('cv2')

        # HardDiskBackend
        img_cv2_color_bgr = mmcv.imread(self.img_path)
        assert img_cv2_color_bgr.shape == (300, 400, 3)
        img_cv2_color_rgb = mmcv.imread(self.img_path, channel_order='rgb')
        assert img_cv2_color_rgb.shape == (300, 400, 3)
        assert_array_equal(img_cv2_color_rgb[:, :, ::-1], img_cv2_color_bgr)
        img_cv2_grayscale1 = mmcv.imread(self.img_path, 'grayscale')
        assert img_cv2_grayscale1.shape == (300, 400)
        img_cv2_grayscale2 = mmcv.imread(self.gray_img_path)
        assert img_cv2_grayscale2.shape == (300, 400, 3)
        img_cv2_unchanged = mmcv.imread(self.gray_img_path, 'unchanged')
        assert img_cv2_unchanged.shape == (300, 400)
        img_cv2_unchanged = mmcv.imread(img_cv2_unchanged)
        assert_array_equal(img_cv2_unchanged, mmcv.imread(img_cv2_unchanged))

        img_cv2_color_bgr = mmcv.imread(self.img_path_obj)
        assert img_cv2_color_bgr.shape == (300, 400, 3)
        img_cv2_color_rgb = mmcv.imread(self.img_path_obj, channel_order='rgb')
        assert img_cv2_color_rgb.shape == (300, 400, 3)
        assert_array_equal(img_cv2_color_rgb[:, :, ::-1], img_cv2_color_bgr)
        img_cv2_grayscale1 = mmcv.imread(self.img_path_obj, 'grayscale')
        assert img_cv2_grayscale1.shape == (300, 400)
        img_cv2_grayscale2 = mmcv.imread(self.gray_img_path_obj)
        assert img_cv2_grayscale2.shape == (300, 400, 3)
        img_cv2_unchanged = mmcv.imread(self.gray_img_path_obj, 'unchanged')
        assert img_cv2_unchanged.shape == (300, 400)
        with pytest.raises(TypeError):
            mmcv.imread(1)

        # PetrelBackend
        img_cv2_color_bgr = mmcv.imread(self.img_path)
        with patch.object(
                PetrelBackend, 'get',
                return_value=img_cv2_color_bgr) as mock_method:
            img_cv2_color_bgr_petrel = mmcv.imread(self.s3_path, backend='cv2')
            img_cv2_color_bgr_petrel_with_args = mmcv.imread(
                self.s3_path,
                backend='cv2',
                file_client_args={'backend': 'petrel'})
            mock_method.assert_called()
            assert_array_equal(img_cv2_color_bgr_petrel,
                               img_cv2_color_bgr_petrel_with_args)

        # HTTPBackend
        img_cv2_color_bgr = mmcv.imread(self.img_path)
        with patch.object(
                HTTPBackend, 'get',
                return_value=img_cv2_color_bgr) as mock_method:
            img_cv2_color_bgr_http = mmcv.imread(self.http_path, backend='cv2')
            img_cv2_color_bgr_http_with_args = mmcv.imread(
                self.http_path,
                backend='cv2',
                file_client_args={'backend': 'http'})
            mock_method.assert_called()
            assert_array_equal(img_cv2_color_bgr_http,
                               img_cv2_color_bgr_http_with_args)

        with pytest.raises(FileNotFoundError):
            mmcv.imread('/not/exists/' + self.img_path)

        # test arg backend pillow
        img_pil_gray_alpha = mmcv.imread(
            self.gray_alpha_img_path, 'grayscale', backend='pillow')
        assert img_pil_gray_alpha.shape == (400, 500)
        mean = img_pil_gray_alpha[300:, 400:].mean()
        assert_allclose(img_pil_gray_alpha[300:, 400:] - mean, 0)
        img_pil_gray_alpha = mmcv.imread(
            self.gray_alpha_img_path, backend='pillow')
        mean = img_pil_gray_alpha[300:, 400:].mean(axis=(0, 1))
        assert_allclose(img_pil_gray_alpha[300:, 400:] - mean, 0)
        assert img_pil_gray_alpha.shape == (400, 500, 3)
        img_pil_gray_alpha = mmcv.imread(
            self.gray_alpha_img_path, 'unchanged', backend='pillow')
        assert img_pil_gray_alpha.shape == (400, 500, 2)
        img_pil_palette = mmcv.imread(
            self.palette_img_path, 'grayscale', backend='pillow')
        assert img_pil_palette.shape == (300, 400)
        img_pil_palette = mmcv.imread(self.palette_img_path, backend='pillow')
        assert img_pil_palette.shape == (300, 400, 3)
        img_pil_palette = mmcv.imread(
            self.palette_img_path, 'unchanged', backend='pillow')
        assert img_pil_palette.shape == (300, 400)

        # backend pillow
        mmcv.use_backend('pillow')
        img_pil_grayscale1 = mmcv.imread(self.img_path, 'grayscale')
        assert img_pil_grayscale1.shape == (300, 400)
        img_pil_gray_alpha = mmcv.imread(self.gray_alpha_img_path, 'grayscale')
        assert img_pil_gray_alpha.shape == (400, 500)
        mean = img_pil_gray_alpha[300:, 400:].mean()
        assert_allclose(img_pil_gray_alpha[300:, 400:] - mean, 0)
        img_pil_gray_alpha = mmcv.imread(self.gray_alpha_img_path)
        mean = img_pil_gray_alpha[300:, 400:].mean(axis=(0, 1))
        assert_allclose(img_pil_gray_alpha[300:, 400:] - mean, 0)
        assert img_pil_gray_alpha.shape == (400, 500, 3)
        img_pil_gray_alpha = mmcv.imread(self.gray_alpha_img_path, 'unchanged')
        assert img_pil_gray_alpha.shape == (400, 500, 2)
        img_pil_palette = mmcv.imread(self.palette_img_path, 'grayscale')
        assert img_pil_palette.shape == (300, 400)
        img_pil_palette = mmcv.imread(self.palette_img_path)
        assert img_pil_palette.shape == (300, 400, 3)
        img_pil_palette = mmcv.imread(self.palette_img_path, 'unchanged')
        assert img_pil_palette.shape == (300, 400)
        img_pil_grayscale2 = mmcv.imread(self.gray_img_path)
        assert img_pil_grayscale2.shape == (300, 400, 3)
        img_pil_unchanged = mmcv.imread(self.gray_img_path, 'unchanged')
        assert img_pil_unchanged.shape == (300, 400)
        img_pil_unchanged = mmcv.imread(img_pil_unchanged)
        assert_array_equal(img_pil_unchanged, mmcv.imread(img_pil_unchanged))

        img_pil_color_bgr = mmcv.imread(self.img_path_obj)
        assert img_pil_color_bgr.shape == (300, 400, 3)
        img_pil_color_rgb = mmcv.imread(self.img_path_obj, channel_order='rgb')
        assert img_pil_color_rgb.shape == (300, 400, 3)
        assert (img_pil_color_rgb == img_cv2_color_rgb).sum() / float(
            img_cv2_color_rgb.size) > 0.5
        assert_array_equal(img_pil_color_rgb[:, :, ::-1], img_pil_color_bgr)
        img_pil_grayscale1 = mmcv.imread(self.img_path_obj, 'grayscale')
        assert img_pil_grayscale1.shape == (300, 400)
        img_pil_grayscale2 = mmcv.imread(self.gray_img_path_obj)
        assert img_pil_grayscale2.shape == (300, 400, 3)
        img_pil_unchanged = mmcv.imread(self.gray_img_path_obj, 'unchanged')
        assert img_pil_unchanged.shape == (300, 400)
        with pytest.raises(TypeError):
            mmcv.imread(1)

        # backend turbojpeg
        mmcv.use_backend('turbojpeg')

        img_turbojpeg_color_bgr = mmcv.imread(self.img_path)
        assert img_turbojpeg_color_bgr.shape == (300, 400, 3)
        assert_array_equal(img_turbojpeg_color_bgr, img_cv2_color_bgr)

        img_turbojpeg_color_rgb = mmcv.imread(
            self.img_path, channel_order='rgb')
        assert img_turbojpeg_color_rgb.shape == (300, 400, 3)
        assert_array_equal(img_turbojpeg_color_rgb, img_cv2_color_rgb)

        with pytest.raises(ValueError):
            mmcv.imread(self.img_path, channel_order='unsupport_order')

        img_turbojpeg_grayscale1 = mmcv.imread(self.img_path, flag='grayscale')
        assert img_turbojpeg_grayscale1.shape == (300, 400)
        assert_array_equal(img_turbojpeg_grayscale1, img_cv2_grayscale1)

        img_turbojpeg_grayscale2 = mmcv.imread(self.gray_img_path)
        assert img_turbojpeg_grayscale2.shape == (300, 400, 3)
        assert_array_equal(img_turbojpeg_grayscale2, img_cv2_grayscale2)

        img_turbojpeg_grayscale2 = mmcv.imread(img_turbojpeg_grayscale2)
        assert_array_equal(img_turbojpeg_grayscale2,
                           mmcv.imread(img_turbojpeg_grayscale2))

        with pytest.raises(ValueError):
            mmcv.imread(self.gray_img_path, 'unchanged')

        with pytest.raises(TypeError):
            mmcv.imread(1)

        with pytest.raises(AssertionError):
            mmcv.use_backend('unsupport_backend')

        with pytest.raises(ValueError):
            mmcv.imread(self.img_path, 'unsupported_backend')

        # backend tifffile, multi channel tiff file(> 4 channels).
        mmcv.use_backend('tifffile')
        img_tifffile = mmcv.imread(self.tiff_path)
        assert img_tifffile.shape == (200, 150, 5)

        mmcv.use_backend('cv2')

        # consistent exif behaviour
        img_cv2_exif = mmcv.imread(self.exif_img_path)
        img_pil_exif = mmcv.imread(self.exif_img_path, backend='pillow')
        assert img_cv2_exif.shape == (400, 300, 3)
        assert img_pil_exif.shape == (400, 300, 3)
        img_cv2_exif_unchanged = mmcv.imread(
            self.exif_img_path, flag='unchanged')
        img_pil_exif_unchanged = mmcv.imread(
            self.exif_img_path, backend='pillow', flag='unchanged')
        assert img_cv2_exif_unchanged.shape == (300, 400, 3)
        assert img_pil_exif_unchanged.shape == (300, 400, 3)
        img_cv2_color_ignore_exif = mmcv.imread(
            self.exif_img_path, flag='color_ignore_orientation')
        img_pil_color_ignore_exif = mmcv.imread(
            self.exif_img_path,
            backend='pillow',
            flag='color_ignore_orientation')
        assert img_cv2_color_ignore_exif.shape == (300, 400, 3)
        assert img_pil_color_ignore_exif.shape == (300, 400, 3)
        img_cv2_grayscale_ignore_exif = mmcv.imread(
            self.exif_img_path, flag='grayscale_ignore_orientation')
        img_pil_grayscale_ignore_exif = mmcv.imread(
            self.exif_img_path,
            backend='pillow',
            flag='grayscale_ignore_orientation')
        assert img_cv2_grayscale_ignore_exif.shape == (300, 400)
        assert img_pil_grayscale_ignore_exif.shape == (300, 400)

    def test_imfrombytes(self):
        # backend cv2, channel order: bgr
        mmcv.use_backend('cv2')
        with open(self.img_path, 'rb') as f:
            img_bytes = f.read()
        img_cv2 = mmcv.imfrombytes(img_bytes)
        assert img_cv2.shape == (300, 400, 3)

        # backend cv2, channel order: rgb
        mmcv.use_backend('cv2')
        with open(self.img_path, 'rb') as f:
            img_bytes = f.read()
        img_rgb_cv2 = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        assert img_rgb_cv2.shape == (300, 400, 3)
        assert_array_equal(img_rgb_cv2, img_cv2[:, :, ::-1])

        # backend cv2, grayscale, decode as 3 channels
        with open(self.gray_img_path, 'rb') as f:
            img_bytes = f.read()
        gray_img_rgb_cv2 = mmcv.imfrombytes(img_bytes)
        assert gray_img_rgb_cv2.shape == (300, 400, 3)

        # backend cv2, grayscale
        with open(self.gray_img_path, 'rb') as f:
            img_bytes = f.read()
        gray_img_cv2 = mmcv.imfrombytes(img_bytes, flag='grayscale')
        assert gray_img_cv2.shape == (300, 400)

        # backend cv2, grayscale dim3
        with open(self.gray_img_dim3_path, 'rb') as f:
            img_bytes = f.read()
        gray_img_dim3_cv2 = mmcv.imfrombytes(img_bytes, flag='grayscale')
        assert gray_img_dim3_cv2.shape == (300, 400)

        # arg backend pillow, channel order: bgr
        with open(self.img_path, 'rb') as f:
            img_bytes = f.read()
        img_pillow = mmcv.imfrombytes(img_bytes, backend='pillow')
        assert img_pillow.shape == (300, 400, 3)
        # Pillow and opencv decoding may not be the same
        assert (img_cv2 == img_pillow).sum() / float(img_cv2.size) > 0.5

        # backend pillow, channel order: bgr
        mmcv.use_backend('pillow')
        with open(self.img_path, 'rb') as f:
            img_bytes = f.read()
        img_pillow = mmcv.imfrombytes(img_bytes)
        assert img_pillow.shape == (300, 400, 3)
        # Pillow and opencv decoding may not be the same
        assert (img_cv2 == img_pillow).sum() / float(img_cv2.size) > 0.5

        # backend turbojpeg, channel order: bgr
        mmcv.use_backend('turbojpeg')
        with open(self.img_path, 'rb') as f:
            img_bytes = f.read()
        img_turbojpeg = mmcv.imfrombytes(img_bytes)
        assert img_turbojpeg.shape == (300, 400, 3)
        assert_array_equal(img_cv2, img_turbojpeg)

        # backend turbojpeg, channel order: rgb
        with open(self.img_path, 'rb') as f:
            img_bytes = f.read()
        img_rgb_turbojpeg = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        assert img_rgb_turbojpeg.shape == (300, 400, 3)
        assert_array_equal(img_rgb_turbojpeg, img_cv2[:, :, ::-1])

        # backend turbojpeg, grayscale, decode as 3 channels
        with open(self.gray_img_path, 'rb') as f:
            img_bytes = f.read()
        gray_img_turbojpeg = mmcv.imfrombytes(img_bytes)
        assert gray_img_turbojpeg.shape == (300, 400, 3)
        assert_array_equal(gray_img_rgb_cv2, gray_img_turbojpeg)

        # backend turbojpeg, grayscale
        with open(self.gray_img_path, 'rb') as f:
            img_bytes = f.read()
        gray_img_turbojpeg = mmcv.imfrombytes(img_bytes, flag='grayscale')
        assert gray_img_turbojpeg.shape == (300, 400)
        assert_array_equal(gray_img_cv2, gray_img_turbojpeg)

        # backend turbojpeg, grayscale dim3
        with open(self.gray_img_dim3_path, 'rb') as f:
            img_bytes = f.read()
        gray_img_dim3_turbojpeg = mmcv.imfrombytes(img_bytes, flag='grayscale')
        assert gray_img_dim3_turbojpeg.shape == (300, 400)
        assert_array_equal(gray_img_dim3_cv2, gray_img_dim3_turbojpeg)

        mmcv.use_backend('cv2')

        with pytest.raises(ValueError):
            with open(self.img_path, 'rb') as f:
                img_bytes = f.read()
            mmcv.imfrombytes(img_bytes, backend='unsupported_backend')

    def test_imwrite(self):
        img = mmcv.imread(self.img_path)
        out_file = osp.join(tempfile.gettempdir(), 'mmcv_test.jpg')
        mmcv.imwrite(img, out_file)
        rewrite_img = mmcv.imread(out_file)
        os.remove(out_file)
        self.assert_img_equal(img, rewrite_img)

        # test petrel client
        with patch.object(
                PetrelBackend, 'put', return_value=None) as mock_method:
            ret = mmcv.imwrite(img, self.s3_path)
            ret_with_args = mmcv.imwrite(
                img, self.s3_path, file_client_args={'backend': 'petrel'})
            assert ret
            assert ret_with_args
            mock_method.assert_called()

        with pytest.raises(cv2.error):
            mmcv.imwrite(img, 'error_file.jppg')

    @patch('mmcv.image.io.TurboJPEG', None)
    def test_no_turbojpeg(self):
        with pytest.raises(ImportError):
            mmcv.use_backend('turbojpeg')

        mmcv.use_backend('cv2')

    @patch('mmcv.image.io.Image', None)
    def test_no_pillow(self):
        with pytest.raises(ImportError):
            mmcv.use_backend('pillow')

        mmcv.use_backend('cv2')
