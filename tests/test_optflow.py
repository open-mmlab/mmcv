# flake8: noqa

import os
import os.path as osp
import tempfile
import time

import mmcv
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_flowread():
    flow_shape = (60, 80, 2)

    # read .flo file
    flow = mmcv.flowread(osp.join(osp.dirname(__file__), 'data/optflow.flo'))
    assert flow.shape == flow_shape

    # pseudo read
    flow_same = mmcv.flowread(flow)
    assert_array_equal(flow, flow_same)

    # read quantized flow concatenated vertically
    flow = mmcv.flowread(osp.join(osp.dirname(__file__),
                                  'data/optflow_concat0.jpg'),
                         quantize=True,
                         denorm=True)
    assert flow.shape == flow_shape

    # read quantized flow concatenated horizontally
    flow = mmcv.flowread(osp.join(osp.dirname(__file__),
                                  'data/optflow_concat1.jpg'),
                         quantize=True,
                         concat_axis=1,
                         denorm=True)
    assert flow.shape == flow_shape

    # test exceptions
    notflow_file = osp.join(osp.dirname(__file__), 'data/color.jpg')
    with pytest.raises(TypeError):
        mmcv.flowread(1)
    with pytest.raises(IOError):
        mmcv.flowread(notflow_file)
    with pytest.raises(IOError):
        mmcv.flowread(notflow_file, quantize=True)
    with pytest.raises(ValueError):
        mmcv.flowread(np.zeros((100, 100, 1)))


def test_flowwrite():
    flow = np.random.rand(100, 100, 2).astype(np.float32)

    # write to a .flo file
    _, filename = tempfile.mkstemp()
    mmcv.flowwrite(flow, filename)
    flow_from_file = mmcv.flowread(filename)
    assert_array_equal(flow, flow_from_file)
    os.remove(filename)

    # write to two .jpg files
    tmp_filename = osp.join(tempfile.gettempdir(), 'mmcv_test_flow.jpg')
    for concat_axis in range(2):
        mmcv.flowwrite(flow,
                       tmp_filename,
                       quantize=True,
                       concat_axis=concat_axis)
        shape = (200, 100) if concat_axis == 0 else (100, 200)
        assert osp.isfile(tmp_filename)
        assert mmcv.imread(tmp_filename, flag='unchanged').shape == shape
        os.remove(tmp_filename)

    # test exceptions
    with pytest.raises(AssertionError):
        mmcv.flowwrite(flow, tmp_filename, quantize=True, concat_axis=2)


def test_quantize_flow():
    flow = (np.random.rand(10, 8, 2).astype(np.float32) - 0.5) * 15
    max_val = 5.0
    dx, dy = mmcv.quantize_flow(flow, max_val=max_val, norm=False)
    ref = np.zeros_like(flow, dtype=np.uint8)
    for i in range(ref.shape[0]):
        for j in range(ref.shape[1]):
            for k in range(ref.shape[2]):
                val = flow[i, j, k] + max_val
                val = min(max(val, 0), 2 * max_val)
                ref[i, j, k] = min(np.floor(255 * val / (2 * max_val)), 254)
    assert_array_equal(dx, ref[..., 0])
    assert_array_equal(dy, ref[..., 1])
    max_val = 0.5
    dx, dy = mmcv.quantize_flow(flow, max_val=max_val, norm=True)
    ref = np.zeros_like(flow, dtype=np.uint8)
    for i in range(ref.shape[0]):
        for j in range(ref.shape[1]):
            for k in range(ref.shape[2]):
                scale = flow.shape[1] if k == 0 else flow.shape[0]
                val = flow[i, j, k] / scale + max_val
                val = min(max(val, 0), 2 * max_val)
                ref[i, j, k] = min(np.floor(255 * val / (2 * max_val)), 254)
    assert_array_equal(dx, ref[..., 0])
    assert_array_equal(dy, ref[..., 1])


def test_dequantize_flow():
    dx = np.random.randint(256, size=(10, 8), dtype=np.uint8)
    dy = np.random.randint(256, size=(10, 8), dtype=np.uint8)
    max_val = 5.0
    flow = mmcv.dequantize_flow(dx, dy, max_val=max_val, denorm=False)
    ref = np.zeros_like(flow, dtype=np.float32)
    for i in range(ref.shape[0]):
        for j in range(ref.shape[1]):
            ref[i, j, 0] = float(dx[i, j] + 0.5) * 2 * max_val / 255 - max_val
            ref[i, j, 1] = float(dy[i, j] + 0.5) * 2 * max_val / 255 - max_val
    assert_array_almost_equal(flow, ref)
    max_val = 0.5
    flow = mmcv.dequantize_flow(dx, dy, max_val=max_val, denorm=True)
    h, w = dx.shape
    ref = np.zeros_like(flow, dtype=np.float32)
    for i in range(ref.shape[0]):
        for j in range(ref.shape[1]):
            ref[i, j, 0] = (float(dx[i, j] + 0.5) * 2 * max_val / 255 -
                            max_val) * w
            ref[i, j, 1] = (float(dy[i, j] + 0.5) * 2 * max_val / 255 -
                            max_val) * h
    assert_array_almost_equal(flow, ref)


def test_flow2rgb():
    flow = np.array([[[0, 0], [0.5, 0.5], [1, 1], [2, 1], [3, np.inf]]],
                    dtype=np.float32)
    flow_img = mmcv.flow2rgb(flow)
    # yapf: disable
    assert_array_almost_equal(
        flow_img,
        np.array([[[1., 1., 1.],
                   [1., 0.826074731, 0.683772236],
                   [1., 0.652149462, 0.367544472],
                   [1., 0.265650552, 5.96046448e-08],
                   [0., 0., 0.]]],
                 dtype=np.float32))
    # yapf: enable


def test_flow_warp():
    def np_flow_warp(flow, img):
        output = np.zeros_like(img, dtype=img.dtype)
        height = flow.shape[0]
        width = flow.shape[1]

        grid = np.indices((height, width)).swapaxes(0, 1).swapaxes(1, 2)
        dx = grid[:, :, 0] + flow[:, :, 1]
        dy = grid[:, :, 1] + flow[:, :, 0]
        sx = np.floor(dx).astype(int)
        sy = np.floor(dy).astype(int)
        valid = (sx >= 0) & (sx < height - 1) & (sy >= 0) & (sy < width - 1)

        output[valid, :] = img[dx[valid].round().astype(int), dy[valid].round(
        ).astype(int), :]

        return output

    dim = 500
    a = np.random.randn(dim, dim, 3) * 10 + 125
    b = np.random.randn(dim, dim, 2) + 2 + 0.2

    c = mmcv.flow_warp(a, b, interpolate_mode='nearest')

    d = np_flow_warp(b, a)

    simple_a = np.zeros((5, 5, 3))
    simple_a[2, 2, 0] = 1
    simple_b = np.ones((5, 5, 2))

    simple_res_c = np.zeros((5, 5, 3))
    simple_res_c[1, 1, 0] = 1

    res_c = mmcv.flow_warp(simple_a, simple_b, interpolate_mode='bilinear')

    assert_array_equal(c, d)
    assert_array_equal(res_c, simple_res_c)


def test_make_color_wheel():
    default_color_wheel = mmcv.make_color_wheel()
    color_wheel = mmcv.make_color_wheel([2, 2, 2, 2, 2, 2])
    # yapf: disable
    assert_array_equal(default_color_wheel, np.array(
        [[1.       , 0.        , 0.        ],
        [1.        , 0.06666667, 0.        ],
        [1.        , 0.13333334, 0.        ],
        [1.        , 0.2       , 0.        ],
        [1.        , 0.26666668, 0.        ],
        [1.        , 0.33333334, 0.        ],
        [1.        , 0.4       , 0.        ],
        [1.        , 0.46666667, 0.        ],
        [1.        , 0.53333336, 0.        ],
        [1.        , 0.6       , 0.        ],
        [1.        , 0.6666667 , 0.        ],
        [1.        , 0.73333335, 0.        ],
        [1.        , 0.8       , 0.        ],
        [1.        , 0.8666667 , 0.        ],
        [1.        , 0.93333334, 0.        ],
        [1.        , 1.        , 0.        ],
        [0.8333333 , 1.        , 0.        ],
        [0.6666667 , 1.        , 0.        ],
        [0.5       , 1.        , 0.        ],
        [0.33333334, 1.        , 0.        ],
        [0.16666667, 1.        , 0.        ],
        [0.        , 1.        , 0.        ],
        [0.        , 1.        , 0.25      ],
        [0.        , 1.        , 0.5       ],
        [0.        , 1.        , 0.75      ],
        [0.        , 1.        , 1.        ],
        [0.        , 0.90909094, 1.        ],
        [0.        , 0.8181818 , 1.        ],
        [0.        , 0.72727275, 1.        ],
        [0.        , 0.6363636 , 1.        ],
        [0.        , 0.54545456, 1.        ],
        [0.        , 0.45454547, 1.        ],
        [0.        , 0.36363637, 1.        ],
        [0.        , 0.27272728, 1.        ],
        [0.        , 0.18181819, 1.        ],
        [0.        , 0.09090909, 1.        ],
        [0.        , 0.        , 1.        ],
        [0.07692308, 0.        , 1.        ],
        [0.15384616, 0.        , 1.        ],
        [0.23076923, 0.        , 1.        ],
        [0.30769232, 0.        , 1.        ],
        [0.3846154 , 0.        , 1.        ],
        [0.46153846, 0.        , 1.        ],
        [0.53846157, 0.        , 1.        ],
        [0.61538464, 0.        , 1.        ],
        [0.6923077 , 0.        , 1.        ],
        [0.7692308 , 0.        , 1.        ],
        [0.84615386, 0.        , 1.        ],
        [0.9230769 , 0.        , 1.        ],
        [1.        , 0.        , 1.        ],
        [1.        , 0.        , 0.8333333 ],
        [1.        , 0.        , 0.6666667 ],
        [1.        , 0.        , 0.5       ],
        [1.        , 0.        , 0.33333334],
        [1.        , 0.        , 0.16666667]], dtype=np.float32))

    assert_array_equal(
        color_wheel,
        np.array([[1., 0. , 0. ],
                 [1. , 0.5, 0. ],
                 [1. , 1. , 0. ],
                 [0.5, 1. , 0. ],
                 [0. , 1. , 0. ],
                 [0. , 1. , 0.5],
                 [0. , 1. , 1. ],
                 [0. , 0.5, 1. ],
                 [0. , 0. , 1. ],
                 [0.5, 0. , 1. ],
                 [1. , 0. , 1. ],
                 [1. , 0. , 0.5]], dtype=np.float32))
    # yapf: enable
