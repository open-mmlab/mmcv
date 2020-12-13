# flake8: noqa
from .tensorrt_utils import (get_tensorrt_op_path, is_tensorrt_plugin_loaded,
                             load_tensorrt_plugin, load_trt_engine, onnx2trt,
                             save_trt_engine)

__all__ = [
    'onnx2trt', 'save_trt_engine', 'load_trt_engine', 'get_tensorrt_op_path',
    'is_tensorrt_plugin_loaded', 'load_tensorrt_plugin'
]
try:
    import torch
except ImportError:
    pass
else:
    from .tensorrt_utils import TRTWraper
    __all__ += ['TRTWraper']
