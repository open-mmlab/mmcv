import os
from functools import partial

import numpy as np
import onnx
import onnxruntime as rt
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
