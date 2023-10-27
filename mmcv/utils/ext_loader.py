# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os
import pkgutil
import warnings
from collections import namedtuple

import torch

if torch.__version__ != 'parrots':

    class ext_ImportError(Exception):
        def __init__(self, arg):
            print(arg)
            print("mmcv is installed incorrectly.")
            print("1. Uninstall existing mmcv in the environment using `pip uninstall mmcv`")
            print("2. Install mmcv-full following the https://mmcv.readthedocs.io/en/latest/get_started/installation.html or https://mmcv.readthedocs.io/en/latest/get_started/build.html")
            print("For more information, see https://github.com/open-mmlab/mmcv/blob/main/docs/en/faq.md")
    
    class undefine_symbol_Error(Exception):
        def __init__(self, arg):
            print(arg)
            print("If those symbols are CUDA/C++ symbols (e.g., libcudart.so or GLIBCXX), check whether the CUDA/GCC runtimes are the same as those used for compiling mmcv")
            print("If those symbols are Pytorch symbols (e.g., symbols containing caffe, aten, and TH), check whether the Pytorch version is the same as that used for compiling mmcv")
            print("For more information, see https://github.com/open-mmlab/mmcv/blob/main/docs/en/faq.md")

    class ext_not_found_Error(Exception):
        def __init__(self, arg):
            print(arg)
            print("mmcv is installed incorrectly.")
            print("1. Uninstall existing mmcv in the environment using `pip uninstall mmcv`")
            print("2. Install mmcv-full following the https://mmcv.readthedocs.io/en/latest/get_started/installation.html or https://mmcv.readthedocs.io/en/latest/get_started/build.html")
            print("For more information, see https://github.com/open-mmlab/mmcv/blob/main/docs/en/faq.md")

    def load_ext(name, funcs):
        import re
        try:
            ext = importlib.import_module('mmcv.' + name)
        except Exception, e:
            exception_inf = str(e)

            pattern = "DLL load failed while importing _ext"     
            if re.search(pattern, exception_inf, re.S):
                raise ext_ImportError(exception_inf)
            
            pattern = "undefined symbol"
            if re.search(pattern, exception_inf, re.S):
                raise undefine_symbol_Error(exception_inf)

            pattern = "No module named 'mmcv._ext'"
            if re.search(pattern, exception_inf, re.S):
                raise ext_not_found_Error(exception_inf)
        
        for fun in funcs:
            assert hasattr(ext, fun), f'{fun} miss in module {name}'
        return ext
else:
    from parrots import extension
    from parrots.base import ParrotsException

    has_return_value_ops = [
        'nms',
        'softnms',
        'nms_match',
        'nms_rotated',
        'top_pool_forward',
        'top_pool_backward',
        'bottom_pool_forward',
        'bottom_pool_backward',
        'left_pool_forward',
        'left_pool_backward',
        'right_pool_forward',
        'right_pool_backward',
        'fused_bias_leakyrelu',
        'upfirdn2d',
        'ms_deform_attn_forward',
        'pixel_group',
        'contour_expand',
        'diff_iou_rotated_sort_vertices_forward',
    ]

    def get_fake_func(name, e):

        def fake_func(*args, **kwargs):
            warnings.warn(f'{name} is not supported in parrots now')
            raise e

        return fake_func

    def load_ext(name, funcs):
        ExtModule = namedtuple('ExtModule', funcs)
        ext_list = []
        lib_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        for fun in funcs:
            try:
                ext_fun = extension.load(fun, name, lib_dir=lib_root)
            except ParrotsException as e:
                if 'No element registered' not in e.message:
                    warnings.warn(e.message)
                ext_fun = get_fake_func(fun, e)
                ext_list.append(ext_fun)
            else:
                if fun in has_return_value_ops:
                    ext_list.append(ext_fun.op)
                else:
                    ext_list.append(ext_fun.op_)
        return ExtModule(*ext_list)


def check_ops_exist() -> bool:
    ext_loader = pkgutil.find_loader('mmcv._ext')
    return ext_loader is not None
