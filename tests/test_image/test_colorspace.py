# Copyright (c) Open-MMLab. All rights reserved.
import cv2
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

import mmcv


def test_bgr2gray():
    in_img = np.random.rand(10, 10, 3).astype(np.float32)
    out_img = mmcv.bgr2gray(in_img)
    computed_gray = (
        in_img[:, :, 0] * 0.114 + in_img[:, :, 1] * 0.587 +
        in_img[:, :, 2] * 0.299)
    assert_array_almost_equal(out_img, computed_gray, decimal=4)
    out_img_3d = mmcv.bgr2gray(in_img, True)
    assert out_img_3d.shape == (10, 10, 1)
    assert_array_almost_equal(out_img_3d[..., 0], out_img, decimal=4)


def test_rgb2gray():
    in_img = np.random.rand(10, 10, 3).astype(np.float32)
    out_img = mmcv.rgb2gray(in_img)
    computed_gray = (
        in_img[:, :, 0] * 0.299 + in_img[:, :, 1] * 0.587 +
        in_img[:, :, 2] * 0.114)
    assert_array_almost_equal(out_img, computed_gray, decimal=4)
    out_img_3d = mmcv.rgb2gray(in_img, True)
    assert out_img_3d.shape == (10, 10, 1)
    assert_array_almost_equal(out_img_3d[..., 0], out_img, decimal=4)


def test_gray2bgr():
    in_img = np.random.rand(10, 10).astype(np.float32)
    out_img = mmcv.gray2bgr(in_img)
    assert out_img.shape == (10, 10, 3)
    for i in range(3):
        assert_array_almost_equal(out_img[..., i], in_img, decimal=4)


def test_gray2rgb():
    in_img = np.random.rand(10, 10).astype(np.float32)
    out_img = mmcv.gray2rgb(in_img)
    assert out_img.shape == (10, 10, 3)
    for i in range(3):
        assert_array_almost_equal(out_img[..., i], in_img, decimal=4)


def test_bgr2rgb():
    in_img = np.random.rand(10, 10, 3).astype(np.float32)
    out_img = mmcv.bgr2rgb(in_img)
    assert out_img.shape == in_img.shape
    assert_array_equal(out_img[..., 0], in_img[..., 2])
    assert_array_equal(out_img[..., 1], in_img[..., 1])
    assert_array_equal(out_img[..., 2], in_img[..., 0])


def test_rgb2bgr():
    in_img = np.random.rand(10, 10, 3).astype(np.float32)
    out_img = mmcv.rgb2bgr(in_img)
    assert out_img.shape == in_img.shape
    assert_array_equal(out_img[..., 0], in_img[..., 2])
    assert_array_equal(out_img[..., 1], in_img[..., 1])
    assert_array_equal(out_img[..., 2], in_img[..., 0])


def test_bgr2hsv():
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


def test_bgr2hls():
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


@pytest.mark.parametrize('src,dst,ref', [('bgr', 'gray', cv2.COLOR_BGR2GRAY),
                                         ('rgb', 'gray', cv2.COLOR_RGB2GRAY),
                                         ('bgr', 'rgb', cv2.COLOR_BGR2RGB),
                                         ('rgb', 'bgr', cv2.COLOR_RGB2BGR),
                                         ('bgr', 'hsv', cv2.COLOR_BGR2HSV),
                                         ('hsv', 'bgr', cv2.COLOR_HSV2BGR),
                                         ('bgr', 'hls', cv2.COLOR_BGR2HLS),
                                         ('hls', 'bgr', cv2.COLOR_HLS2BGR)])
def test_imconvert(src, dst, ref):
    img = np.random.rand(10, 10, 3).astype(np.float32)
    assert_array_equal(mmcv.imconvert(img, src, dst), cv2.cvtColor(img, ref))
