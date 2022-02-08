import os
import warnings
from functools import partial

import numpy as np
import onnx
import onnxruntime as rt
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

onnx_file = 'tmp.onnx'


@pytest.fixture(autouse=True)
def run_before_and_after_test():
    # clear onnx_file before test
    if os.path.exists(onnx_file):
        os.remove(onnx_file)

    yield

    # clear onnx_file after test
    if os.path.exists(onnx_file):
        os.remove(onnx_file)


class WrapFunction(nn.Module):

    def __init__(self, wrapped_function):
        super(WrapFunction, self).__init__()
        self.wrapped_function = wrapped_function

    def forward(self, *args, **kwargs):
        return self.wrapped_function(*args, **kwargs)


def process_grid_sample(func, input, grid, ort_custom_op_path=''):
    wrapped_model = WrapFunction(func).eval()

    input_names = ['input', 'grid']
    output_names = ['output']

    with torch.no_grad():
        torch.onnx.export(
            wrapped_model, (input, grid),
            onnx_file,
            export_params=True,
            keep_initializers_as_inputs=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=11)

    onnx_model = onnx.load(onnx_file)

    session_options = rt.SessionOptions()
    if ort_custom_op_path:
        session_options.register_custom_ops_library(ort_custom_op_path)

    # get onnx output
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert (len(net_feed_input) == 2)
    sess = rt.InferenceSession(onnx_file, session_options)
    ort_result = sess.run(None, {
        'input': input.detach().numpy(),
        'grid': grid.detach().numpy()
    })
    pytorch_results = wrapped_model(input.clone(), grid.clone())
    assert np.allclose(pytorch_results, ort_result, atol=1e-3)


@pytest.mark.parametrize('mode', ['bilinear', 'nearest'])
@pytest.mark.parametrize('padding_mode', ['zeros', 'border', 'reflection'])
@pytest.mark.parametrize('align_corners', [True, False])
def test_grid_sample(mode, padding_mode, align_corners):
    from mmcv.onnx.symbolic import register_extra_symbolics
    opset_version = 11
    register_extra_symbolics(opset_version)

    from mmcv.ops import get_onnxruntime_op_path
    ort_custom_op_path = get_onnxruntime_op_path()
    if not os.path.exists(ort_custom_op_path):
        pytest.skip('custom ops for onnxruntime are not compiled.')

    input = torch.rand(1, 1, 10, 10)
    grid = torch.Tensor([[[1, 0, 0], [0, 1, 0]]])
    grid = F.affine_grid(
        grid, (1, 1, 15, 15), align_corners=align_corners).type_as(input)

    def func(input, grid):
        return F.grid_sample(
            input,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners)

    return process_grid_sample(func, input, grid, ort_custom_op_path)


@pytest.mark.parametrize('align_corners', [True, False])
def test_bilinear_grid_sample(align_corners):
    from mmcv.ops.point_sample import bilinear_grid_sample

    # only support pytorch >= 1.5.0
    if version.parse(torch.__version__) < version.parse('1.5.0'):
        pytest.skip('Only support PyTorch >= 1.5.0')

    input = torch.rand(1, 1, 10, 10)
    grid = torch.Tensor([[[1, 0, 0], [0, 1, 0]]])
    grid = F.affine_grid(
        grid, (1, 1, 15, 15), align_corners=align_corners).type_as(input)

    def func(input, grid):
        return bilinear_grid_sample(input, grid, align_corners=align_corners)

    return process_grid_sample(func, input, grid)


def test_nms():
    if torch.__version__ == 'parrots':
        pytest.skip('onnx is not supported in parrots directly')
    from mmcv.ops import get_onnxruntime_op_path, nms
    np_boxes = np.array([[6.0, 3.0, 8.0, 7.0], [3.0, 6.0, 9.0, 11.0],
                         [3.0, 7.0, 10.0, 12.0], [1.0, 4.0, 13.0, 7.0]],
                        dtype=np.float32)
    np_scores = np.array([0.6, 0.9, 0.7, 0.2], dtype=np.float32)
    boxes = torch.from_numpy(np_boxes)
    scores = torch.from_numpy(np_scores)

    nms = partial(
        nms, iou_threshold=0.3, offset=0, score_threshold=0, max_num=0)
    pytorch_dets, _ = nms(boxes, scores)
    pytorch_score = pytorch_dets[:, 4]

    wrapped_model = WrapFunction(nms)
    wrapped_model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model, (boxes, scores),
            onnx_file,
            export_params=True,
            keep_initializers_as_inputs=True,
            input_names=['boxes', 'scores'],
            opset_version=11)

    onnx_model = onnx.load(onnx_file)
    ort_custom_op_path = get_onnxruntime_op_path()
    session_options = rt.SessionOptions()
    if os.path.exists(ort_custom_op_path):
        session_options.register_custom_ops_library(ort_custom_op_path)

    # get onnx output
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert (len(net_feed_input) == 2)
    sess = rt.InferenceSession(onnx_file, session_options)
    onnx_dets, _ = sess.run(None, {
        'scores': scores.detach().numpy(),
        'boxes': boxes.detach().numpy()
    })
    onnx_score = onnx_dets[:, 4]
    assert np.allclose(pytorch_score, onnx_score, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires GPU')
def test_softnms():
    if torch.__version__ == 'parrots':
        pytest.skip('onnx is not supported in parrots directly')
    from mmcv.ops import get_onnxruntime_op_path, soft_nms

    # only support pytorch >= 1.7.0
    if version.parse(torch.__version__) < version.parse('1.7.0'):
        warnings.warn('test_softnms should be ran with pytorch >= 1.7.0')
        return

    # only support onnxruntime >= 1.5.1
    assert version.parse(rt.__version__) >= version.parse(
        '1.5.1'), 'test_softnms should be ran with onnxruntime >= 1.5.1'

    ort_custom_op_path = get_onnxruntime_op_path()
    if not os.path.exists(ort_custom_op_path):
        pytest.skip('softnms for onnxruntime is not compiled.')

    np_boxes = np.array([[6.0, 3.0, 8.0, 7.0], [3.0, 6.0, 9.0, 11.0],
                         [3.0, 7.0, 10.0, 12.0], [1.0, 4.0, 13.0, 7.0]],
                        dtype=np.float32)
    np_scores = np.array([0.6, 0.9, 0.7, 0.2], dtype=np.float32)

    boxes = torch.from_numpy(np_boxes)
    scores = torch.from_numpy(np_scores)

    configs = [[0.3, 0.5, 0.01, 'linear'], [0.3, 0.5, 0.01, 'gaussian'],
               [0.3, 0.5, 0.01, 'naive']]

    session_options = rt.SessionOptions()
    session_options.register_custom_ops_library(ort_custom_op_path)

    for _iou_threshold, _sigma, _min_score, _method in configs:
        pytorch_dets, pytorch_inds = soft_nms(
            boxes,
            scores,
            iou_threshold=_iou_threshold,
            sigma=_sigma,
            min_score=_min_score,
            method=_method)
        nms = partial(
            soft_nms,
            iou_threshold=_iou_threshold,
            sigma=_sigma,
            min_score=_min_score,
            method=_method)

        wrapped_model = WrapFunction(nms)
        wrapped_model.cpu().eval()
        with torch.no_grad():
            torch.onnx.export(
                wrapped_model, (boxes, scores),
                onnx_file,
                export_params=True,
                keep_initializers_as_inputs=True,
                input_names=['boxes', 'scores'],
                opset_version=11)
        onnx_model = onnx.load(onnx_file)

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 2)
        sess = rt.InferenceSession(onnx_file, session_options)
        onnx_dets, onnx_inds = sess.run(None, {
            'scores': scores.detach().numpy(),
            'boxes': boxes.detach().numpy()
        })

        assert np.allclose(pytorch_dets, onnx_dets, atol=1e-3)
        assert np.allclose(onnx_inds, onnx_inds, atol=1e-3)


def test_roialign():
    if torch.__version__ == 'parrots':
        pytest.skip('onnx is not supported in parrots directly')
    try:
        from mmcv.ops import get_onnxruntime_op_path, roi_align
    except (ImportError, ModuleNotFoundError):
        pytest.skip('roi_align op is not successfully compiled')

    ort_custom_op_path = get_onnxruntime_op_path()
    # roi align config
    pool_h = 2
    pool_w = 2
    spatial_scale = 1.0
    sampling_ratio = 2

    inputs = [([[[[1., 2.], [3., 4.]]]], [[0., 0., 0., 1., 1.]]),
              ([[[[1., 2.], [3., 4.]], [[4., 3.],
                                        [2., 1.]]]], [[0., 0., 0., 1., 1.]]),
              ([[[[1., 2., 5., 6.], [3., 4., 7., 8.], [9., 10., 13., 14.],
                  [11., 12., 15., 16.]]]], [[0., 0., 0., 3., 3.]])]

    def warpped_function(torch_input, torch_rois):
        return roi_align(torch_input, torch_rois, (pool_w, pool_h),
                         spatial_scale, sampling_ratio, 'avg', True)

    for case in inputs:
        np_input = np.array(case[0], dtype=np.float32)
        np_rois = np.array(case[1], dtype=np.float32)
        input = torch.from_numpy(np_input)
        rois = torch.from_numpy(np_rois)

        # compute pytorch_output
        with torch.no_grad():
            pytorch_output = roi_align(input, rois, (pool_w, pool_h),
                                       spatial_scale, sampling_ratio, 'avg',
                                       True)

        # export and load onnx model
        wrapped_model = WrapFunction(warpped_function)
        with torch.no_grad():
            torch.onnx.export(
                wrapped_model, (input, rois),
                onnx_file,
                export_params=True,
                keep_initializers_as_inputs=True,
                input_names=['input', 'rois'],
                opset_version=11)

        onnx_model = onnx.load(onnx_file)
        session_options = rt.SessionOptions()
        if os.path.exists(ort_custom_op_path):
            session_options.register_custom_ops_library(ort_custom_op_path)

        # compute onnx_output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 2)
        sess = rt.InferenceSession(onnx_file, session_options)
        onnx_output = sess.run(None, {
            'input': input.detach().numpy(),
            'rois': rois.detach().numpy()
        })
        onnx_output = onnx_output[0]

        # allclose

        assert np.allclose(pytorch_output, onnx_output, atol=1e-3)


def test_roialign_rotated():
    if torch.__version__ == 'parrots':
        pytest.skip('onnx is not supported in parrots directly')
    try:
        from mmcv.ops import get_onnxruntime_op_path, roi_align_rotated
    except (ImportError, ModuleNotFoundError):
        pytest.skip('roi_align_aligned op is not successfully compiled')

    ort_custom_op_path = get_onnxruntime_op_path()
    if not os.path.exists(ort_custom_op_path):
        pytest.skip('custom ops for onnxruntime are not compiled.')
    # roi align config
    pool_h = 2
    pool_w = 2
    spatial_scale = 1.0
    sampling_ratio = 2

    inputs = [([[[[1., 2.], [3., 4.]]]], [[0., 0.5, 0.5, 1., 1., 0]]),
              ([[[[1., 2.], [3., 4.]]]], [[0., 0.5, 0.5, 1., 1., np.pi / 2]]),
              ([[[[1., 2.], [3., 4.]],
                 [[4., 3.], [2., 1.]]]], [[0., 0.5, 0.5, 1., 1., 0]]),
              ([[[[1., 2., 5., 6.], [3., 4., 7., 8.], [9., 10., 13., 14.],
                  [11., 12., 15., 16.]]]], [[0., 1.5, 1.5, 3., 3., 0]]),
              ([[[[1., 2., 5., 6.], [3., 4., 7., 8.], [9., 10., 13., 14.],
                  [11., 12., 15., 16.]]]], [[0., 1.5, 1.5, 3., 3.,
                                             np.pi / 2]])]

    def warpped_function(torch_input, torch_rois):
        return roi_align_rotated(torch_input, torch_rois, (pool_w, pool_h),
                                 spatial_scale, sampling_ratio, True, False)

    for case in inputs:
        np_input = np.array(case[0], dtype=np.float32)
        np_rois = np.array(case[1], dtype=np.float32)
        input = torch.from_numpy(np_input)
        rois = torch.from_numpy(np_rois)

        # compute pytorch_output
        with torch.no_grad():
            pytorch_output = roi_align_rotated(input, rois, (pool_w, pool_h),
                                               spatial_scale, sampling_ratio,
                                               True, False)

        # export and load onnx model
        wrapped_model = WrapFunction(warpped_function)
        with torch.no_grad():
            torch.onnx.export(
                wrapped_model, (input, rois),
                onnx_file,
                export_params=True,
                keep_initializers_as_inputs=True,
                input_names=['features', 'rois'],
                opset_version=11)

        onnx_model = onnx.load(onnx_file)
        session_options = rt.SessionOptions()
        if os.path.exists(ort_custom_op_path):
            session_options.register_custom_ops_library(ort_custom_op_path)

        # compute onnx_output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 2)
        sess = rt.InferenceSession(onnx_file, session_options)
        onnx_output = sess.run(None, {
            'features': input.detach().numpy(),
            'rois': rois.detach().numpy()
        })
        onnx_output = onnx_output[0]

        # allclose

        assert np.allclose(pytorch_output, onnx_output, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires GPU')
def test_roipool():
    if torch.__version__ == 'parrots':
        pytest.skip('onnx is not supported in parrots directly')
    from mmcv.ops import roi_pool

    # roi pool config
    pool_h = 2
    pool_w = 2
    spatial_scale = 1.0

    inputs = [([[[[1., 2.], [3., 4.]]]], [[0., 0., 0., 1., 1.]]),
              ([[[[1., 2.], [3., 4.]], [[4., 3.],
                                        [2., 1.]]]], [[0., 0., 0., 1., 1.]]),
              ([[[[1., 2., 5., 6.], [3., 4., 7., 8.], [9., 10., 13., 14.],
                  [11., 12., 15., 16.]]]], [[0., 0., 0., 3., 3.]])]

    def warpped_function(torch_input, torch_rois):
        return roi_pool(torch_input, torch_rois, (pool_w, pool_h),
                        spatial_scale)

    for case in inputs:
        np_input = np.array(case[0], dtype=np.float32)
        np_rois = np.array(case[1], dtype=np.float32)
        input = torch.from_numpy(np_input).cuda()
        rois = torch.from_numpy(np_rois).cuda()

        # compute pytorch_output
        with torch.no_grad():
            pytorch_output = roi_pool(input, rois, (pool_w, pool_h),
                                      spatial_scale)
            pytorch_output = pytorch_output.cpu()

        # export and load onnx model
        wrapped_model = WrapFunction(warpped_function)
        with torch.no_grad():
            torch.onnx.export(
                wrapped_model, (input, rois),
                onnx_file,
                export_params=True,
                keep_initializers_as_inputs=True,
                input_names=['input', 'rois'],
                opset_version=11)
        onnx_model = onnx.load(onnx_file)

        # compute onnx_output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 2)
        sess = rt.InferenceSession(onnx_file)
        onnx_output = sess.run(
            None, {
                'input': input.detach().cpu().numpy(),
                'rois': rois.detach().cpu().numpy()
            })
        onnx_output = onnx_output[0]

        # allclose
        assert np.allclose(pytorch_output, onnx_output, atol=1e-3)


def test_interpolate():
    from mmcv.onnx.symbolic import register_extra_symbolics
    opset_version = 11
    register_extra_symbolics(opset_version)

    def func(feat, scale_factor=2):
        out = F.interpolate(feat, scale_factor=scale_factor)
        return out

    net = WrapFunction(func)
    net = net.cpu().eval()
    dummy_input = torch.randn(2, 4, 8, 8).cpu()
    torch.onnx.export(
        net,
        dummy_input,
        onnx_file,
        input_names=['input'],
        opset_version=opset_version)
    sess = rt.InferenceSession(onnx_file)
    onnx_result = sess.run(None, {'input': dummy_input.detach().numpy()})
    pytorch_result = func(dummy_input).detach().numpy()

    assert np.allclose(pytorch_result, onnx_result, atol=1e-3)


@pytest.mark.parametrize('mode', ['top', 'bottom', 'left', 'right'])
def test_corner_pool(mode, opset=11):
    if torch.__version__ == 'parrots':
        pytest.skip('onnx is not supported in parrots directly')

    from mmcv.ops import get_onnxruntime_op_path
    ort_custom_op_path = get_onnxruntime_op_path()
    if not os.path.exists(ort_custom_op_path):
        pytest.skip('custom ops for onnxruntime are not compiled.')

    from mmcv.ops.corner_pool import CornerPool

    def corner_pool_func(input):
        corner_pool_module = CornerPool(mode)
        return corner_pool_module.corner_pool.apply(input)

    wrapped_model = WrapFunction(corner_pool_func).eval()

    input = torch.rand((2, 3, 9, 12))  # (n,c,h,w)

    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            input,
            onnx_file,
            export_params=True,
            keep_initializers_as_inputs=True,
            input_names=['input'],
            output_names=['output'],
            opset_version=opset)

    onnx_model = onnx.load(onnx_file)
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert (len(net_feed_input) == 1)

    session_options = rt.SessionOptions()
    session_options.register_custom_ops_library(ort_custom_op_path)
    sess = rt.InferenceSession(onnx_file, session_options)
    ort_result = sess.run(None, {'input': input.detach().numpy()})
    pytorch_results = wrapped_model(input.clone())

    assert np.allclose(pytorch_results, ort_result, atol=1e-5)


@pytest.mark.parametrize('key', ['cummax', 'cummin'])
def test_cummax_cummin(key, opset=11):
    if torch.__version__ == 'parrots':
        pytest.skip('onnx is not supported in parrots directly')

    # Note generally `cummax` or `cummin` is exportable to ONNX
    # as long as the pytorch version >= 1.5.0, since `torch.cummax`
    # is only supported with torch >= 1.5.0.
    # But when `cummax` or `cummin` serves as an intermediate component
    # whose outputs is used as inputs for another modules, it's expected
    # that pytorch version must be >= 1.7.0. Otherwise error appears like:
    # `RuntimeError: tuple  appears in op that does not forward tuples,
    # unsupported 'kind: prim::PythonOp`.
    if version.parse(torch.__version__) < version.parse('1.7.0'):
        pytest.skip('test_cummax_cummin should be ran with pytorch >= 1.7.0')

    # register custom op `mmcv::cummax` and `mmcv::cummin`
    from mmcv.onnx.symbolic import register_extra_symbolics
    register_extra_symbolics(opset)

    from mmcv.ops import get_onnxruntime_op_path
    ort_custom_op_path = get_onnxruntime_op_path()
    if not os.path.exists(ort_custom_op_path):
        pytest.skip('custom ops for onnxruntime are not compiled.')

    input_list = [
        # arbitrary shape, e.g. 1-D, 2-D, 3-D, ...
        torch.rand((2, 3, 4, 1, 5)),
        torch.rand((1)),
        torch.rand((2, 0, 1)),  # tensor.numel() is 0
        torch.FloatTensor(),  # empty tensor
    ]

    cummax_cummin_funcs = {'cummax': torch.cummax, 'cummin': torch.cummin}

    for input in input_list:
        ndims = input.dim()
        # valid dim range is [-ndims, ndims-1]
        # test for all `dim` value which is valid
        for dim in range(-ndims, ndims):
            cummax_func = partial(cummax_cummin_funcs[key], dim=dim)
            wrapped_model = WrapFunction(cummax_func).eval()

            with torch.no_grad():
                torch.onnx.export(
                    wrapped_model,
                    input,
                    onnx_file,
                    export_params=True,
                    keep_initializers_as_inputs=True,
                    input_names=['input'],
                    output_names=['output', 'indices'],
                    opset_version=opset)

            onnx_model = onnx.load(onnx_file)
            input_all = [node.name for node in onnx_model.graph.input]
            input_initializer = [
                node.name for node in onnx_model.graph.initializer
            ]
            net_feed_input = list(set(input_all) - set(input_initializer))
            assert (len(net_feed_input) == 1)

            session_options = rt.SessionOptions()
            session_options.register_custom_ops_library(ort_custom_op_path)
            sess = rt.InferenceSession(onnx_file, session_options)
            ort_output, ort_inds = sess.run(None,
                                            {'input': input.detach().numpy()})
            pytorch_output, pytorch_inds = wrapped_model(input.clone())
            pytorch_output = pytorch_output.detach().numpy()
            pytorch_inds = pytorch_inds.detach().numpy()
            assert np.allclose(pytorch_output, ort_output, atol=1e-5)
            assert np.all(pytorch_inds == ort_inds)


@pytest.mark.parametrize('shifts_dims_pair', [([-3, 5], [2, 0]), (5, None)])
def test_roll(shifts_dims_pair):
    opset = 11
    from mmcv.onnx.symbolic import register_extra_symbolics
    register_extra_symbolics(opset)

    input = torch.arange(0, 4 * 5 * 6, dtype=torch.float32).view(4, 5, 6)

    shifts, dims = shifts_dims_pair
    func = partial(torch.roll, shifts=shifts, dims=dims)
    wrapped_model = WrapFunction(func).eval()

    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            input,
            onnx_file,
            export_params=True,
            keep_initializers_as_inputs=True,
            input_names=['input'],
            output_names=['output'],
            opset_version=opset)

    onnx_model = onnx.load(onnx_file)
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert (len(net_feed_input) == 1)

    sess = rt.InferenceSession(onnx_file)
    ort_output = sess.run(None, {'input': input.detach().numpy()})[0]

    with torch.no_grad():
        pytorch_output = wrapped_model(input.clone())

    torch.testing.assert_allclose(ort_output, pytorch_output)


@pytest.mark.skipif(
    torch.__version__ == 'parrots',
    reason='onnx is not supported in parrots directly')
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='modulated_deform_conv2d only supports in GPU')
def test_modulated_deform_conv2d():
    try:
        from mmcv.ops import ModulatedDeformConv2d, get_onnxruntime_op_path
    except (ImportError, ModuleNotFoundError):
        pytest.skip('modulated_deform_conv op is not successfully compiled')

    ort_custom_op_path = get_onnxruntime_op_path()
    if not os.path.exists(ort_custom_op_path):
        pytest.skip('custom ops for onnxruntime are not compiled.')

    # modulated deform conv config
    in_channels = 3
    out_channels = 64
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    deform_groups = 1
    kernel_size = 3

    input = torch.rand(1, in_channels, 28, 28).cuda()  # (n, c, h, w)
    conv_offset = nn.Conv2d(
        in_channels=3,
        out_channels=deform_groups * 3 * kernel_size * kernel_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True).cuda()
    conv_offset.cuda()
    out = conv_offset(input)
    o1, o2, mask = torch.chunk(out, 3, dim=1)
    offset = torch.cat((o1, o2), dim=1)
    mask = torch.sigmoid(mask)

    model_with_bias = ModulatedDeformConv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        deform_groups,
        bias=True)
    model_without_bias = ModulatedDeformConv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        deform_groups,
        bias=False)
    models = [model_with_bias.cuda(), model_without_bias.cuda()]

    for model in models:
        # export and load onnx model
        with torch.no_grad():
            torch.onnx.export(
                model, (input, offset, mask),
                onnx_file,
                export_params=True,
                keep_initializers_as_inputs=True,
                input_names=['input', 'offset', 'mask'],
                opset_version=11)

        session_options = rt.SessionOptions()
        if os.path.exists(ort_custom_op_path):
            session_options.register_custom_ops_library(ort_custom_op_path)

        # compute onnx_output
        sess = rt.InferenceSession(onnx_file, session_options)
        onnx_output = sess.run(
            None, {
                'input': input.cpu().detach().numpy(),
                'offset': offset.cpu().detach().numpy(),
                'mask': mask.cpu().detach().numpy()
            })[0]

        # compute pytorch_output
        with torch.no_grad():
            pytorch_output = model(input, offset, mask).cpu()
        # allclose
        assert np.allclose(pytorch_output, onnx_output, atol=1e-3)


@pytest.mark.skipif(
    torch.__version__ == 'parrots',
    reason='onnx is not supported in parrots directly')
def test_deform_conv2d(threshold=1e-3):
    try:
        from mmcv.ops import DeformConv2d, get_onnxruntime_op_path
    except (ImportError, ModuleNotFoundError):
        pytest.skip('deform_conv op is not successfully compiled')

    ort_custom_op_path = get_onnxruntime_op_path()
    if not os.path.exists(ort_custom_op_path):
        pytest.skip('custom ops for onnxruntime are not compiled.')

    # deform conv config
    # modulated deform conv config
    in_channels = 1
    out_channels = 64
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    deform_groups = 1
    kernel_size = 2
    input = [[[[1., 2., 3.], [0., 1., 2.], [3., 5., 2.]]]]
    offset_weight = [[[0.1, 0.4, 0.6, 0.1]], [[0.3, 0.2, 0.1, 0.3]],
                     [[0.5, 0.5, 0.2, 0.8]], [[0.8, 0.3, 0.9, 0.1]],
                     [[0.3, 0.1, 0.2, 0.5]], [[0.3, 0.7, 0.5, 0.3]],
                     [[0.6, 0.2, 0.5, 0.3]], [[0.4, 0.1, 0.8, 0.4]]]
    offset_bias = [0.7, 0.1, 0.8, 0.5, 0.6, 0.5, 0.4, 0.7]
    deform_weight = [[[0.4, 0.2, 0.1, 0.9]]]

    x = torch.tensor(input)
    conv_offset = nn.Conv2d(
        in_channels=in_channels,
        out_channels=deform_groups * 2 * kernel_size * kernel_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True)

    conv_offset.weight.data = torch.nn.Parameter(
        torch.Tensor(offset_weight).reshape(8, 1, 2, 2))
    conv_offset.bias.data = torch.nn.Parameter(
        torch.Tensor(offset_bias).reshape(8))

    offset = conv_offset(x)

    model = DeformConv2d(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, deform_groups)

    model.weight.data = torch.nn.Parameter(
        torch.Tensor(deform_weight).reshape(1, 1, 2, 2))

    with torch.no_grad():
        torch.onnx.export(
            model, (x, offset),
            onnx_file,
            export_params=True,
            keep_initializers_as_inputs=True,
            input_names=['input', 'offset'],
            opset_version=11)

    session_options = rt.SessionOptions()
    if os.path.exists(ort_custom_op_path):
        session_options.register_custom_ops_library(ort_custom_op_path)

    # compute onnx_output
    sess = rt.InferenceSession(onnx_file, session_options)
    onnx_output = sess.run(
        None, {
            'input': x.cpu().detach().numpy(),
            'offset': offset.cpu().detach().numpy(),
        })[0]

    # compute pytorch_output
    with torch.no_grad():
        pytorch_output = model(x, offset).cpu()
    # allclose
    assert np.allclose(pytorch_output, onnx_output, atol=1e-3)
