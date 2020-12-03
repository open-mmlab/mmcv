import numpy as np
import onnx
import os
import pytest
import torch

try:
    from mmcv.utils import (TRTWraper, load_tensorrt_plugin, onnx2trt,
                            save_trt_engine)
except:
    pytest.skip('test requires tensorrt')

onnx_file = 'tmp.onnx'
trt_file = 'tmp.engine'


def test_roialign():
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('test requires GPU')
    try:
        from mmcv.ops import RoIAlign
    except ModuleNotFoundError:
        pytest.skip('test requires compilation')

    load_tensorrt_plugin()

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
            torch.onnx.export(wrapped_model, (input, rois),
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
        trt_engine = onnx2trt(onnx_model,
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
