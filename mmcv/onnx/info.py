import os

from ..tensorrt import is_tensorrt_plugin_loaded


def is_custom_op_loaded():
    flag = False
    if is_tensorrt_plugin_loaded():
        flag = True
    else:
        try:
            from ..ops import get_onnxruntime_op_path
            ort_lib_path = get_onnxruntime_op_path()
            flag = os.path.exists(ort_lib_path)
        except (ImportError, ModuleNotFoundError):
            pass
    return flag
