# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os
import pkgutil
import warnings
from collections import namedtuple

import torch


def load_ext(name, funcs):
    """Load extensions from module 'mmcv.{name}'.
    
    If the module is not found, returns a MockExtModule that will raise
    NotImplementedError when any of the functions are called.
    """
    class MockExtModule:
        """Mock extension module to use when CUDA extensions are not available."""
        def __init__(self, funcs):
            self.funcs = funcs
            
        def __getattr__(self, name):
            if name in self.funcs:
                msg = (f"Function '{name}' is not available. CUDA extensions are not installed. "
                       f"Using PyTorch-only fallbacks where available.")
                def not_implemented(*args, **kwargs):
                    raise NotImplementedError(msg)
                return not_implemented
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    try:
        ext = importlib.import_module('mmcv.' + name)
        # Check that requested functions are available
        for fun in funcs:
            assert hasattr(ext, fun), f'{fun} miss in module {name}'
        return ext
    except (ImportError, ModuleNotFoundError):
        # If the module is not found, return a mock module
        # This allows the code to continue running and possibly use alternative pure PyTorch implementations
        warnings.warn(f"Module 'mmcv.{name}' not found. CUDA extensions are not installed. "
                     f"Using PyTorch-only fallbacks where available.")
        return MockExtModule(funcs)


def check_ops_exist() -> bool:
    """Check if compiled operations are available."""
    ext_loader = pkgutil.find_loader('mmcv._ext')
    return ext_loader is not None
