import os
from functools import partial

import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn

onnx_file = 'tmp.onnx'
trt_file = 'tmp.engine'


class WrapFunction(nn.Module):

    def __init__(self, wrapped_function):
        super(WrapFunction, self).__init__()
        self.wrapped_function = wrapped_function

    def forward(self, *args, **kwargs):
        return self.wrapped_function(*args, **kwargs)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is required for test_roialign')
def test_roialign():
    try:
        from mmcv.tensorrt import (TRTWraper, onnx2trt, save_trt_engine,
                                   is_tensorrt_plugin_loaded)

        if not is_tensorrt_plugin_loaded:
            pytest.skip('test requires to complie TensorRT plugins in mmcv')
    except (ImportError, ModuleNotFoundError):
        pytest.skip('test requires mmcv.tensorrt')

    try:
        from mmcv.ops import RoIAlign
    except (ImportError, ModuleNotFoundError):
        pytest.skip('test requires compilation')

    # trt config
    fp16_mode = False
    max_workspace_size = 1 << 30

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

    wrapped_model = RoIAlign((pool_w, pool_h), spatial_scale, sampling_ratio,
                             'avg', True).cuda()
    for case in inputs:
        np_input = np.array(case[0], dtype=np.float32)
        np_rois = np.array(case[1], dtype=np.float32)
        input = torch.from_numpy(np_input).cuda()
        rois = torch.from_numpy(np_rois).cuda()

        with torch.no_grad():
            torch.onnx.export(
                wrapped_model, (input, rois),
                onnx_file,
                export_params=True,
                keep_initializers_as_inputs=True,
                input_names=['input', 'rois'],
                output_names=['roi_feat'],
                opset_version=11)
        onnx_model = onnx.load(onnx_file)

        # create trt engine and wraper
        opt_shape_dict = {
            'input': [list(input.shape),
                      list(input.shape),
                      list(input.shape)],
            'rois': [list(rois.shape),
                     list(rois.shape),
                     list(rois.shape)]
        }
        trt_engine = onnx2trt(
            onnx_model,
            opt_shape_dict,
            fp16_mode=fp16_mode,
            max_workspace_size=max_workspace_size)
        save_trt_engine(trt_engine, trt_file)
        trt_model = TRTWraper(trt_file, ['input', 'rois'], ['roi_feat'])

        with torch.no_grad():
            trt_outputs = trt_model({'input': input, 'rois': rois})
            trt_roi_feat = trt_outputs['roi_feat']
            trt_roi_feat = trt_roi_feat.cpu().detach().numpy()

        # compute pytorch_output
        with torch.no_grad():
            pytorch_roi_feat = wrapped_model(input, rois)
            pytorch_roi_feat = pytorch_roi_feat.cpu().detach().numpy()

        # allclose
        if os.path.exists(onnx_file):
            os.remove(onnx_file)
        if os.path.exists(trt_file):
            os.remove(trt_file)
        assert np.allclose(pytorch_roi_feat, trt_roi_feat, atol=1e-3)


def test_nms():
    try:
        from mmcv.tensorrt import (TRTWraper, onnx2trt, save_trt_engine,
                                   is_tensorrt_plugin_loaded)

        if not is_tensorrt_plugin_loaded:
            pytest.skip('test requires to complie TensorRT plugins in mmcv')
    except (ImportError, ModuleNotFoundError):
        pytest.skip('test requires mmcv.tensorrt')

    try:
        from mmcv.ops import nms
    except (ImportError, ModuleNotFoundError):
        pytest.skip('test requires compilation')

    # trt config
    fp16_mode = False
    max_workspace_size = 1 << 30

    np_boxes = np.array([[6.0, 3.0, 8.0, 7.0], [3.0, 6.0, 9.0, 11.0],
                         [3.0, 7.0, 10.0, 12.0], [1.0, 4.0, 13.0, 7.0]],
                        dtype=np.float32)
    np_scores = np.array([0.6, 0.9, 0.7, 0.2], dtype=np.float32)
    boxes = torch.from_numpy(np_boxes).cuda()
    scores = torch.from_numpy(np_scores).cuda()
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
            output_names=['dets', 'inds'],
            opset_version=11)
    onnx_model = onnx.load(onnx_file)

    # create trt engine and wraper
    opt_shape_dict = {
        'boxes': [list(boxes.shape),
                  list(boxes.shape),
                  list(boxes.shape)],
        'scores': [list(scores.shape),
                   list(scores.shape),
                   list(scores.shape)]
    }
    trt_engine = onnx2trt(
        onnx_model,
        opt_shape_dict,
        fp16_mode=fp16_mode,
        max_workspace_size=max_workspace_size)
    save_trt_engine(trt_engine, trt_file)
    trt_model = TRTWraper(trt_file, ['boxes', 'scores'], ['dets', 'inds'])

    with torch.no_grad():
        trt_outputs = trt_model({'boxes': boxes, 'scores': scores})
        trt_dets = trt_outputs['dets']
        trt_dets = trt_dets.cpu().detach().numpy()

    # compute pytorch_output
    with torch.no_grad():
        pytorch_outputs = wrapped_model(boxes, scores)
        pytorch_dets = pytorch_outputs[0].cpu().detach().numpy()

    # allclose
    if os.path.exists(onnx_file):
        os.remove(onnx_file)
    if os.path.exists(trt_file):
        os.remove(trt_file)
    num_boxes = pytorch_dets.shape[0]
    trt_dets = trt_dets[:num_boxes, ...]
    trt_scores = trt_dets[:, 4]
    pytorch_scores = pytorch_dets[:, 4]
    assert np.allclose(pytorch_scores, trt_scores, atol=1e-3)
