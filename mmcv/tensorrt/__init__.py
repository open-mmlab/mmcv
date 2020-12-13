from .tensorrt_utils import (onnx2trt, save_trt_engine, load_trt_engine,
                             get_tensorrt_op_path, is_tensorrt_plugin_loaded,
                             load_tensorrt_plugin)
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
