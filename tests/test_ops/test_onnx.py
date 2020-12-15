import os
import warnings
from functools import partial

import numpy as np
import onnx
import onnxruntime as rt
import pytest
import torch
import torch.nn as nn

onnx_file = 'tmp.onnx'


class WrapFunction(nn.Module):

    def __init__(self, wrapped_function):
        super(WrapFunction, self).__init__()
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
    pytorch_dets, _ = nms(boxes, scores, iou_threshold=0.3, offset=0)
    pytorch_score = pytorch_dets[:, 4]
    nms = partial(nms, iou_threshold=0.3, offset=0)
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
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert (len(net_feed_input) == 2)
    sess = rt.InferenceSession(onnx_file)
    onnx_dets, _ = sess.run(None, {
        'scores': scores.detach().numpy(),
        'boxes': boxes.detach().numpy()
    })
    onnx_score = onnx_dets[:, 4]
    os.remove(onnx_file)
    assert np.allclose(pytorch_score, onnx_score, atol=1e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='CUDA is unavailable for test_softnms')
def test_softnms():
    from mmcv.ops import get_onnxruntime_op_path, soft_nms
    from packaging import version

    # only support pytorch >= 1.7.0
    if version.parse(torch.__version__) < version.parse('1.7.0'):
        warnings.warn('test_softnms should be ran with pytorch >= 1.7.0')
        return

    # only support onnxruntime >= 1.5.1
    assert version.parse(rt.__version__) >= version.parse(
        '1.5.1'), 'test_softnms should be ran with onnxruntime >= 1.5.1'

    ort_custom_op_path = get_onnxruntime_op_path()
    assert os.path.exists(ort_custom_op_path)

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
        os.remove(onnx_file)
        assert np.allclose(pytorch_dets, onnx_dets, atol=1e-3)
        assert np.allclose(onnx_inds, onnx_inds, atol=1e-3)


def test_roialign():
    from mmcv.ops import roi_align

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

        # compute onnx_output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 2)
        sess = rt.InferenceSession(onnx_file)
        onnx_output = sess.run(None, {
            'input': input.detach().numpy(),
            'rois': rois.detach().numpy()
        })
        onnx_output = onnx_output[0]

        # allclose
        os.remove(onnx_file)
        assert np.allclose(pytorch_output, onnx_output, atol=1e-3)


def test_roipool():
    if not torch.cuda.is_available():
        return
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
        os.remove(onnx_file)
        assert np.allclose(pytorch_output, onnx_output, atol=1e-3)
