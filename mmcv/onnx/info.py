import os


def is_custom_op_loaded():
    flag = False
    try:
        from ..tensorrt import is_tensorrt_plugin_loaded
        flag = is_tensorrt_plugin_loaded()
    except (ImportError, ModuleNotFoundError):
        pass
    if not flag:
        try:
            from ..ops import get_onnxruntime_op_path
            ort_lib_path = get_onnxruntime_op_path()
            flag = os.path.exists(ort_lib_path)
        except (ImportError, ModuleNotFoundError):
            pass
    return flag
