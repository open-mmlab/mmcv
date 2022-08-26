# Copyright (c) OpenMMLab. All rights reserved.
import os
from functools import partial

import numpy as np
import onnx
import onnxruntime as rt
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

onnx_file = 'tmp.onnx'
if torch.__version__ == 'parrots':
    pytest.skip('not supported in parrots now', allow_module_level=True)


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
        super().__init__()
        self.wrapped_function = wrapped_function

    def forward(self, *args, **kwargs):
        return self.wrapped_function(*args, **kwargs)


def test_nms():
    from mmcv.ops import nms
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
    session_options = rt.SessionOptions()

    # get onnx output
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert (len(net_feed_input) == 2)
    sess = rt.InferenceSession(
        onnx_file, session_options, providers=['CPUExecutionProvider'])
    onnx_dets, _ = sess.run(None, {
        'scores': scores.detach().numpy(),
        'boxes': boxes.detach().numpy()
    })
    onnx_score = onnx_dets[:, 4]
    assert np.allclose(pytorch_score, onnx_score, atol=1e-3)


def test_roialign():
    try:
        from mmcv.ops import roi_align
    except (ImportError, ModuleNotFoundError):
        pytest.skip('roi_align op is not successfully compiled')

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

        # compute onnx_output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 2)
        sess = rt.InferenceSession(
            onnx_file, session_options, providers=['CPUExecutionProvider'])
        onnx_output = sess.run(None, {
            'input': input.detach().numpy(),
            'rois': rois.detach().numpy()
        })
        onnx_output = onnx_output[0]

        # allclose

        assert np.allclose(pytorch_output, onnx_output, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires GPU')
def test_roipool():
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
        sess = rt.InferenceSession(
            onnx_file, providers=['CPUExecutionProvider'])
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
    sess = rt.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
    onnx_result = sess.run(None, {'input': dummy_input.detach().numpy()})
    pytorch_result = func(dummy_input).detach().numpy()

    assert np.allclose(pytorch_result, onnx_result, atol=1e-3)


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

    sess = rt.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
    ort_output = sess.run(None, {'input': input.detach().numpy()})[0]

    with torch.no_grad():
        pytorch_output = wrapped_model(input.clone())

    torch.testing.assert_allclose(ort_output, pytorch_output)


def _test_symbolic(model, inputs, symbol_name):
    with torch.no_grad():
        torch.onnx.export(model, inputs, onnx_file, opset_version=11)

    import onnx
    model = onnx.load(onnx_file)
    nodes = model.graph.node

    symbol_exist = False
    for n in nodes:
        if n.op_type == symbol_name:
            symbol_exist = True
    assert symbol_exist


@pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires GPU')
def test_border_align():
    from mmcv.ops import BorderAlign
    model = BorderAlign(2)
    input = torch.rand(1, 8, 2, 2).cuda()
    boxes = torch.rand(1, 4, 4).cuda()
    _test_symbolic(model, (input, boxes), 'MMCVBorderAlign')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires GPU')
def test_carafe():
    from mmcv.ops import CARAFENaive
    feat = torch.randn(2, 64, 3, 3, device='cuda').double()
    mask = torch.randn(2, 100, 6, 6, device='cuda').sigmoid().double()
    _test_symbolic(CARAFENaive(5, 4, 2), (feat, mask), 'MMCVCARAFENaive')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires GPU')
def test_deform_conv():
    from mmcv.ops import DeformConv2dPack
    x = torch.randn(1, 2, 4, 4, device='cuda')
    _test_symbolic(
        DeformConv2dPack(2, 4, 3, 1, 1).cuda(), x, 'MMCVDeformConv2d')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires GPU')
def test_modulated_deform_conv():
    from mmcv.ops import ModulatedDeformConv2dPack
    x = torch.randn(1, 2, 4, 4, device='cuda')
    _test_symbolic(
        ModulatedDeformConv2dPack(2, 4, 3, 1, 1).cuda(), x,
        'MMCVModulatedDeformConv2d')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires GPU')
def test_deform_roi_pool():
    from mmcv.ops import DeformRoIPoolPack
    x = torch.tensor([[[[1., 2.], [3., 4.]]]], device='cuda')
    rois = torch.tensor([[0., 0., 0., 1., 1.]], device='cuda')
    output_c = x.size(1)
    pool_h = 2
    pool_w = 2
    spatial_scale = 1.0
    sampling_ratio = 2
    model = DeformRoIPoolPack((pool_h, pool_w),
                              output_c,
                              spatial_scale=spatial_scale,
                              sampling_ratio=sampling_ratio).cuda()

    _test_symbolic(model, (x, rois), 'MMCVDeformRoIPool')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires GPU')
def test_masked_conv():
    from mmcv.ops import MaskedConv2d
    x = torch.rand(1, 2, 4, 4, device='cuda')
    mask = torch.rand(1, 4, 4, device='cuda')
    _test_symbolic(
        MaskedConv2d(2, 4, 3, 1, 1).cuda(), (x, mask), 'MMCVMaskedConv2d')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires GPU')
def test_pr_roi_pool():
    from mmcv.ops import PrRoIPool
    pool_h = 2
    pool_w = 2
    spatial_scale = 1.0
    x = torch.tensor([[[[1., 2.], [3., 4.]]]], device='cuda')
    rois = torch.tensor([[0., 0., 0., 1., 1.]], device='cuda')
    model = PrRoIPool((pool_h, pool_w), spatial_scale).cuda()
    _test_symbolic(model, (x, rois), 'PrRoIPool')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires GPU')
def test_psa_mask():
    from mmcv.ops import PSAMask
    input = torch.rand(4, 16, 8, 8).cuda()
    model = PSAMask('collect', (4, 4)).cuda()
    _test_symbolic(model, input, 'MMCVPSAMask')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires GPU')
def test_roi_align_rotated():
    from mmcv.ops import RoIAlignRotated
    pool_h = 2
    pool_w = 2
    spatial_scale = 1.0
    sampling_ratio = 2
    x = torch.tensor([[[[1., 2.], [3., 4.]]]], device='cuda')
    rois = torch.tensor([[0., 0.5, 0.5, 1., 1., 0]], device='cuda')
    model = RoIAlignRotated((pool_h, pool_w), spatial_scale,
                            sampling_ratio).cuda()
    _test_symbolic(model, (x, rois), 'MMCVRoIAlignRotated')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires GPU')
def test_roi_feaeture_align():
    from mmcv.ops import rotated_feature_align
    wrapped_model = WrapFunction(rotated_feature_align)
    feature = torch.rand(1, 1, 2, 2, device='cuda')
    bbox = torch.rand(1, 2, 2, 5, device='cuda')
    _test_symbolic(wrapped_model, (feature, bbox), 'MMCVRotatedFeatureAlign')
