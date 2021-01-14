import os

import numpy as np
import onnx
import pytest
import torch

from mmcv.tensorrt import (TRTWraper, is_tensorrt_plugin_loaded, onnx2trt,
                           save_trt_engine)

onnx_file = 'tmp.onnx'
trt_file = 'tmp.engine'

with_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is required for test_roialign')

with_tensorrt = pytest.mark.skipif(
    not is_tensorrt_plugin_loaded(),
    reason='test requires to complie TensorRT plugins in mmcv')


class WrapFunction(torch.nn.Module):

    def __init__(self, wrapped_function):
        super(WrapFunction, self).__init__()
        self.wrapped_function = wrapped_function

    def forward(self, *args, **kwargs):
        return self.wrapped_function(*args, **kwargs)


@with_cuda
@with_tensorrt
def test_roialign():
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

        # compute pytorch_output
        with torch.no_grad():
            pytorch_roi_feat = wrapped_model(input, rois)

        # allclose
        if os.path.exists(onnx_file):
            os.remove(onnx_file)
        if os.path.exists(trt_file):
            os.remove(trt_file)
        assert torch.allclose(pytorch_roi_feat, trt_roi_feat)


@with_cuda
@with_tensorrt
def test_scatternd():

    def func(data):
        data[:, :-2] += 1
        data[:2, :] -= 1
        return data

    data = torch.zeros(4, 4).cuda()
    wrapped_model = WrapFunction(func).eval().cuda()

    input_names = ['input']
    output_names = ['output']

    with torch.no_grad():
        torch.onnx.export(
            wrapped_model, (data.clone(), ),
            onnx_file,
            export_params=True,
            keep_initializers_as_inputs=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=11)

    onnx_model = onnx.load(onnx_file)

    # create trt engine and wraper
    opt_shape_dict = {
        'input': [list(data.shape),
                  list(data.shape),
                  list(data.shape)],
    }
    # trt config
    fp16_mode = False
    max_workspace_size = 1 << 30

    trt_engine = onnx2trt(
        onnx_model,
        opt_shape_dict,
        fp16_mode=fp16_mode,
        max_workspace_size=max_workspace_size)

    save_trt_engine(trt_engine, trt_file)
    trt_model = TRTWraper(trt_file, input_names, output_names)

    with torch.no_grad():
        trt_outputs = trt_model({'input': data.clone()})
        trt_results = trt_outputs['output']

    # compute pytorch_output
    with torch.no_grad():
        pytorch_results = wrapped_model(data.clone())

    # allclose
    if os.path.exists(onnx_file):
        os.remove(onnx_file)
    if os.path.exists(trt_file):
        os.remove(trt_file)
    assert torch.allclose(pytorch_results, trt_results)
