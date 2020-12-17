import importlib
import os
import pkgutil
from collections import namedtuple

import torch

has_return_value_ops = [
    'nms', 'softnms', 'nms_match', 'top_pool_forward', 'top_pool_backward',
    'bottom_pool_forward', 'bottom_pool_backward', 'left_pool_forward',
    'left_pool_backward', 'right_pool_forward', 'right_pool_backward'
]

if torch.__version__ != 'parrots':

    def load_ext(name, funcs):
        ext = importlib.import_module('mmcv.' + name)
        for fun in funcs:
            assert hasattr(ext, fun), f'{fun} miss in module {name}'
        return ext
else:
    from parrots import extension

    def get_extension(ext_str, root = ''):
        import os
        from ctypes import cdll
        if os.path.isfile(ext_str):
            lib_path = ext_str
        else:
            from distutils.command.build_ext import build_ext
            lib = build_ext.get_ext_filename(None, ext_str)
            lib_path = os.path.join(root, lib)
        ext = cdll.LoadLibrary(lib_path)
        return ext

    def check_func_in_extension(func, ext):
        exist = True
        try:
            getattr(ext, func)
        except AttributeError:
            exist = False
        return exist

    def load_ext(name, funcs):
        ExtModule = namedtuple('ExtModule', funcs)
        ext_list = []
        lib_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        loaded_ext = get_extension(name, lib_root)
        for fun in funcs:
            if check_func_in_extension(fun, loaded_ext):
                ext = extension.load(fun, name, lib_dir=lib_root)
            else:
                ext = extension.load(fun, '_ext_pt', lib_dir=lib_root)
            if fun in has_return_value_ops:
                # op : out = func(in, **attrs_dict)
                ext_list.append(ext.op)
            else:
                # op_ : func(in, out, **attrs_dict)
                ext_list.append(ext.op_)
        return ExtModule(*ext_list)


def check_ops_exist():
    ext_loader = pkgutil.find_loader('mmcv._ext')
    return ext_loader is not None
