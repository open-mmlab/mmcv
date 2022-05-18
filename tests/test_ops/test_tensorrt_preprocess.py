# Copyright (c) OpenMMLab. All rights reserved.
import os
from functools import wraps

import onnx
import pytest
import torch

from mmcv.ops import nms
from mmcv.tensorrt.preprocess import preprocess_onnx

if torch.__version__ == 'parrots':
    pytest.skip('not supported in parrots now', allow_module_level=True)


def remove_tmp_file(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        onnx_file = 'tmp.onnx'
        kwargs['onnx_file'] = onnx_file
        try:
            result = func(*args, **kwargs)
        finally:
            if os.path.exists(onnx_file):
                os.remove(onnx_file)
        return result

    return wrapper


@remove_tmp_file
def export_nms_module_to_onnx(module, onnx_file):
    torch_model = module()
    torch_model.eval()

    input = (torch.rand([100, 4], dtype=torch.float32),
             torch.rand([100], dtype=torch.float32))

    torch.onnx.export(
        torch_model,
        input,
        onnx_file,
        opset_version=11,
        input_names=['boxes', 'scores'],
        output_names=['output'])

    onnx_model = onnx.load(onnx_file)
    return onnx_model


def test_can_handle_nms_with_constant_maxnum():

    class ModuleNMS(torch.nn.Module):

        def forward(self, boxes, scores):
            return nms(boxes, scores, iou_threshold=0.4, max_num=10)

    onnx_model = export_nms_module_to_onnx(ModuleNMS)
    preprocess_onnx_model = preprocess_onnx(onnx_model)
    for node in preprocess_onnx_model.graph.node:
        if 'NonMaxSuppression' in node.name:
            assert len(node.attribute) == 5, 'The NMS must have 5 attributes.'


def test_can_handle_nms_with_undefined_maxnum():

    class ModuleNMS(torch.nn.Module):

        def forward(self, boxes, scores):
            return nms(boxes, scores, iou_threshold=0.4)

    onnx_model = export_nms_module_to_onnx(ModuleNMS)
    preprocess_onnx_model = preprocess_onnx(onnx_model)
    for node in preprocess_onnx_model.graph.node:
        if 'NonMaxSuppression' in node.name:
            assert len(node.attribute) == 5, \
                'The NMS must have 5 attributes.'
            assert node.attribute[2].i > 0, \
                'The max_output_boxes_per_class is not defined correctly.'
