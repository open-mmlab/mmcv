from __future__ import division

import numpy as np
import pytest

import mmcv


def test_quantize():
    arr = np.random.randn(10, 10)
    levels = 20

    qarr = mmcv.quantize(arr, -1, 1, levels)
    assert qarr.shape == arr.shape
    assert qarr.dtype == np.dtype('int64')
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ref = min(levels - 1,
                      int(np.floor(10 * (1 + max(min(arr[i, j], 1), -1)))))
            assert qarr[i, j] == ref

    qarr = mmcv.quantize(arr, -1, 1, 20, dtype=np.uint8)
    assert qarr.shape == arr.shape
    assert qarr.dtype == np.dtype('uint8')

    with pytest.raises(ValueError):
        mmcv.quantize(arr, -1, 1, levels=0)
    with pytest.raises(ValueError):
        mmcv.quantize(arr, -1, 1, levels=10.0)
    with pytest.raises(ValueError):
        mmcv.quantize(arr, 2, 1, levels)


def test_dequantize():
    levels = 20
    qarr = np.random.randint(levels, size=(10, 10))

    arr = mmcv.dequantize(qarr, -1, 1, levels)
    assert arr.shape == qarr.shape
    assert arr.dtype == np.dtype('float64')
    for i in range(qarr.shape[0]):
        for j in range(qarr.shape[1]):
            assert arr[i, j] == (qarr[i, j] + 0.5) / 10 - 1

    arr = mmcv.dequantize(qarr, -1, 1, levels, dtype=np.float32)
    assert arr.shape == qarr.shape
    assert arr.dtype == np.dtype('float32')

    with pytest.raises(ValueError):
        mmcv.dequantize(arr, -1, 1, levels=0)
    with pytest.raises(ValueError):
        mmcv.dequantize(arr, -1, 1, levels=10.0)
    with pytest.raises(ValueError):
        mmcv.dequantize(arr, 2, 1, levels)


def test_joint():
    arr = np.random.randn(100, 100)
    levels = 1000
    qarr = mmcv.quantize(arr, -1, 1, levels)
    recover = mmcv.dequantize(qarr, -1, 1, levels)
    assert np.abs(recover[arr < -1] + 0.999).max() < 1e-6
    assert np.abs(recover[arr > 1] - 0.999).max() < 1e-6
    assert np.abs((recover - arr)[(arr >= -1) & (arr <= 1)]).max() <= 1e-3

    arr = np.clip(np.random.randn(100) / 1000, -0.01, 0.01)
    levels = 99
    qarr = mmcv.quantize(arr, -1, 1, levels)
    recover = mmcv.dequantize(qarr, -1, 1, levels)
    assert np.all(recover == 0)
