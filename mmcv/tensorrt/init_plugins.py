import ctypes
import glob
import os


def get_tensorrt_op_path():
    """Get TensorRT plugins library path."""
    wildcard = os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
        '_ext_trt.*.so')

    paths = glob.glob(wildcard)
    lib_path = paths[0] if len(paths) > 0 else ''
    return lib_path


plugin_is_loaded = False


def is_tensorrt_plugin_loaded():
    """Check if TensorRT plugins library is loaded or not.

    Returns:
        bool: plugin_is_loaded flag
    """
    global plugin_is_loaded
    return plugin_is_loaded


def load_tensorrt_plugin():
    """load TensorRT plugins library."""
    global plugin_is_loaded
    lib_path = get_tensorrt_op_path()
    if (not plugin_is_loaded) and os.path.exists(lib_path):
        ctypes.CDLL(lib_path)
        plugin_is_loaded = True
