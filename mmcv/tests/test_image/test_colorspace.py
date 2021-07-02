# Copyright (c) Open-MMLab. All rights reserved.
import cv2
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

import mmcv
from mmcv.image.colorspace import (_convert_input_type_range,
                                   _convert_output_type_range)


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
    computed_hsv = np.empty_like(in_img)
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            b, g, r = in_img[i, j]
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


def test_convert_input_type_range():
    with pytest.raises(TypeError):
        # The img type should be np.float32 or np.uint8
        in_img = np.random.rand(10, 10, 3).astype(np.uint64)
        _convert_input_type_range(in_img)
    # np.float32
    in_img = np.random.rand(10, 10, 3).astype(np.float32)
    out_img = _convert_input_type_range(in_img)
    assert out_img.dtype == np.float32
    assert np.absolute(out_img).mean() < 1
    # np.uint8
    in_img = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    out_img = _convert_input_type_range(in_img)
    assert out_img.dtype == np.float32
    assert np.absolute(out_img).mean() < 1


def test_convert_output_type_range():
    with pytest.raises(TypeError):
        # The dst_type should be np.float32 or np.uint8
        in_img = np.random.rand(10, 10, 3).astype(np.float32)
        _convert_output_type_range(in_img, np.uint64)
    # np.float32
    in_img = (np.random.rand(10, 10, 3) * 255).astype(np.float32)
    out_img = _convert_output_type_range(in_img, np.float32)
    assert out_img.dtype == np.float32
    assert np.absolute(out_img).mean() < 1
    # np.uint8
    in_img = (np.random.rand(10, 10, 3) * 255).astype(np.float32)
    out_img = _convert_output_type_range(in_img, np.uint8)
    assert out_img.dtype == np.uint8
    assert np.absolute(out_img).mean() > 1


def assert_image_almost_equal(x, y, atol=1):
    assert x.dtype == np.uint8
    assert y.dtype == np.uint8
    assert np.all(np.abs(x.astype(np.int32) - y.astype(np.int32)) <= atol)


def test_rgb2ycbcr():
    with pytest.raises(TypeError):
        # The img type should be np.float32 or np.uint8
        in_img = np.random.rand(10, 10, 3).astype(np.uint64)
        mmcv.rgb2ycbcr(in_img)

    # float32
    in_img = np.random.rand(10, 10, 3).astype(np.float32)
    out_img = mmcv.rgb2ycbcr(in_img)
    computed_ycbcr = np.empty_like(in_img)
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            r, g, b = in_img[i, j]
            y = 16 + r * 65.481 + g * 128.553 + b * 24.966
            cb = 128 - r * 37.797 - g * 74.203 + b * 112.0
            cr = 128 + r * 112.0 - g * 93.786 - b * 18.214
            computed_ycbcr[i, j, :] = [y, cb, cr]
    computed_ycbcr /= 255.
    assert_array_almost_equal(out_img, computed_ycbcr, decimal=2)
    # y_only=True
    out_img = mmcv.rgb2ycbcr(in_img, y_only=True)
    computed_y = np.empty_like(out_img, dtype=out_img.dtype)
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            r, g, b = in_img[i, j]
            y = 16 + r * 65.481 + g * 128.553 + b * 24.966
            computed_y[i, j] = y
    computed_y /= 255.
    assert_array_almost_equal(out_img, computed_y, decimal=2)

    # uint8
    in_img = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    out_img = mmcv.rgb2ycbcr(in_img)
    computed_ycbcr = np.empty_like(in_img)
    in_img = in_img / 255.
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            r, g, b = in_img[i, j]
            y = 16 + r * 65.481 + g * 128.553 + b * 24.966
            cb = 128 - r * 37.797 - g * 74.203 + b * 112.0
            cr = 128 + r * 112.0 - g * 93.786 - b * 18.214
            y, cb, cr = y.round(), cb.round(), cr.round()
            computed_ycbcr[i, j, :] = [y, cb, cr]
    assert_image_almost_equal(out_img, computed_ycbcr)
    # y_only=True
    in_img = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    out_img = mmcv.rgb2ycbcr(in_img, y_only=True)
    computed_y = np.empty_like(out_img, dtype=out_img.dtype)
    in_img = in_img / 255.
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            r, g, b = in_img[i, j]
            y = 16 + r * 65.481 + g * 128.553 + b * 24.966
            y = y.round()
            computed_y[i, j] = y
    assert_image_almost_equal(out_img, computed_y)


def test_bgr2ycbcr():
    # float32
    in_img = np.random.rand(10, 10, 3).astype(np.float32)
    out_img = mmcv.bgr2ycbcr(in_img)
    computed_ycbcr = np.empty_like(in_img)
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            b, g, r = in_img[i, j]
            y = 16 + r * 65.481 + g * 128.553 + b * 24.966
            cb = 128 - r * 37.797 - g * 74.203 + b * 112.0
            cr = 128 + r * 112.0 - g * 93.786 - b * 18.214
            computed_ycbcr[i, j, :] = [y, cb, cr]
    computed_ycbcr /= 255.
    assert_array_almost_equal(out_img, computed_ycbcr, decimal=2)
    # y_only=True
    in_img = np.random.rand(10, 10, 3).astype(np.float32)
    out_img = mmcv.bgr2ycbcr(in_img, y_only=True)
    computed_y = np.empty_like(out_img, dtype=out_img.dtype)
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            b, g, r = in_img[i, j]
            y = 16 + r * 65.481 + g * 128.553 + b * 24.966
            computed_y[i, j] = y
    computed_y /= 255.
    assert_array_almost_equal(out_img, computed_y, decimal=2)

    # uint8
    in_img = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    out_img = mmcv.bgr2ycbcr(in_img)
    computed_ycbcr = np.empty_like(in_img)
    in_img = in_img / 255.
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            b, g, r = in_img[i, j]
            y = 16 + r * 65.481 + g * 128.553 + b * 24.966
            cb = 128 - r * 37.797 - g * 74.203 + b * 112.0
            cr = 128 + r * 112.0 - g * 93.786 - b * 18.214
            y, cb, cr = y.round(), cb.round(), cr.round()
            computed_ycbcr[i, j, :] = [y, cb, cr]
    assert_image_almost_equal(out_img, computed_ycbcr)
    # y_only = True
    in_img = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    out_img = mmcv.bgr2ycbcr(in_img, y_only=True)
    computed_y = np.empty_like(out_img, dtype=out_img.dtype)
    in_img = in_img / 255.
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            b, g, r = in_img[i, j]
            y = 16 + r * 65.481 + g * 128.553 + b * 24.966
            y = y.round()
            computed_y[i, j] = y
    assert_image_almost_equal(out_img, computed_y)


def test_ycbcr2rgb():
    with pytest.raises(TypeError):
        # The img type should be np.float32 or np.uint8
        in_img = np.random.rand(10, 10, 3).astype(np.uint64)
        mmcv.ycbcr2rgb(in_img)

    # float32
    in_img = np.random.rand(10, 10, 3).astype(np.float32)
    out_img = mmcv.ycbcr2rgb(in_img)
    computed_rgb = np.empty_like(in_img)
    in_img *= 255.
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            y, cb, cr = in_img[i, j]
            r = -222.921 + y * 0.00456621 * 255 + cr * 0.00625893 * 255
            g = 135.576 + y * 0.00456621 * 255 - cb * 0.00153632 * 255 - \
                cr * 0.00318811 * 255
            b = -276.836 + y * 0.00456621 * 255. + cb * 0.00791071 * 255
            computed_rgb[i, j, :] = [r, g, b]
    computed_rgb /= 255.
    assert_array_almost_equal(out_img, computed_rgb, decimal=2)

    # uint8
    in_img = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    out_img = mmcv.ycbcr2rgb(in_img)
    computed_rgb = np.empty_like(in_img)
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            y, cb, cr = in_img[i, j]
            r = -222.921 + y * 0.00456621 * 255 + cr * 0.00625893 * 255
            g = 135.576 + y * 0.00456621 * 255 - cb * 0.00153632 * 255 - \
                cr * 0.00318811 * 255
            b = -276.836 + y * 0.00456621 * 255. + cb * 0.00791071 * 255
            r, g, b = r.round(), g.round(), b.round()
            computed_rgb[i, j, :] = [r, g, b]
    assert_image_almost_equal(out_img, computed_rgb)


def test_ycbcr2bgr():
    # float32
    in_img = np.random.rand(10, 10, 3).astype(np.float32)
    out_img = mmcv.ycbcr2bgr(in_img)
    computed_bgr = np.empty_like(in_img)
    in_img *= 255.
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            y, cb, cr = in_img[i, j]
            r = -222.921 + y * 0.00456621 * 255 + cr * 0.00625893 * 255
            g = 135.576 + y * 0.00456621 * 255 - cb * 0.00153632 * 255 - \
                cr * 0.00318811 * 255
            b = -276.836 + y * 0.00456621 * 255. + cb * 0.00791071 * 255
            computed_bgr[i, j, :] = [b, g, r]
    computed_bgr /= 255.
    assert_array_almost_equal(out_img, computed_bgr, decimal=2)

    # uint8
    in_img = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    out_img = mmcv.ycbcr2bgr(in_img)
    computed_bgr = np.empty_like(in_img)
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            y, cb, cr = in_img[i, j]
            r = -222.921 + y * 0.00456621 * 255 + cr * 0.00625893 * 255
            g = 135.576 + y * 0.00456621 * 255 - cb * 0.00153632 * 255 - \
                cr * 0.00318811 * 255
            b = -276.836 + y * 0.00456621 * 255. + cb * 0.00791071 * 255
            r, g, b = r.round(), g.round(), b.round()
            computed_bgr[i, j, :] = [b, g, r]
    assert_image_almost_equal(out_img, computed_bgr)


def test_bgr2hls():
    in_img = np.random.rand(10, 10, 3).astype(np.float32)
    out_img = mmcv.bgr2hls(in_img)
    argmax = in_img.argmax(axis=2)
    computed_hls = np.empty_like(in_img)
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            b, g, r = in_img[i, j]
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
