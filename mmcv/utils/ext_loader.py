# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os
import pkgutil
import warnings
from collections import namedtuple
import inspect

import torch


def load_ext(name, funcs):
    """Load extensions from module 'mmcv.{name}'.
    
    In the PyTorch-only version, we don't rely on compiled extensions.
    Instead, this function either:
    1. Returns a module with pure PyTorch implementations if they exist
    2. Returns a MockExtModule that will warn when functions without PyTorch 
       implementations are called.
    """
    class MockExtModule:
        """Mock extension module that uses PyTorch-only implementations."""
        def __init__(self, funcs):
            self.funcs = funcs
            self._pytorch_fallbacks = {}
            self._load_pytorch_fallbacks()
            
        def _load_pytorch_fallbacks(self):
            """Load pure PyTorch implementations for the functions."""
            # Try to import any pure PyTorch implementations
            try:
                if 'nms' in self.funcs or 'softnms' in self.funcs or 'nms_match' in self.funcs:
                    from mmcv.ops.pure_pytorch_nms import (
                        nms_pytorch, soft_nms_pytorch, nms_match_pytorch, nms_quadri_pytorch
                    )
                    self._pytorch_fallbacks['nms'] = nms_pytorch
                    self._pytorch_fallbacks['softnms'] = soft_nms_pytorch
                    self._pytorch_fallbacks['nms_match'] = nms_match_pytorch
                    self._pytorch_fallbacks['nms_quadri'] = nms_quadri_pytorch
                
                if 'roi_align_forward' in self.funcs or 'roi_pool_forward' in self.funcs:
                    from mmcv.ops.pure_pytorch_roi import roi_align_pytorch, roi_pool_pytorch
                    self._pytorch_fallbacks['roi_align_forward'] = roi_align_pytorch
                    self._pytorch_fallbacks['roi_pool_forward'] = roi_pool_pytorch
                    
                    # Add dummy backward implementations that warn users
                    def backward_warning(*args, **kwargs):
                        warnings.warn("Backward operation is not fully implemented in PyTorch-only mode. "
                                     "This may affect training but should work for inference.")
                        return torch.zeros_like(args[0])
                    
                    self._pytorch_fallbacks['roi_align_backward'] = backward_warning
                    self._pytorch_fallbacks['roi_pool_backward'] = backward_warning
                
                # Add more fallback implementations as they are created
                
            except ImportError as e:
                warnings.warn(f"Failed to import some PyTorch-only implementations: {e}")
            
        def __getattr__(self, name):
            # Check if we have a PyTorch implementation for this function
            if name in self._pytorch_fallbacks:
                return self._pytorch_fallbacks[name]
                
            # Otherwise, warn the user
            if name in self.funcs:
                msg = (f"Function '{name}' has no PyTorch-only implementation yet. "
                       f"This operation will fail in PyTorch-only mode.")
                warnings.warn(msg)
                
                def not_implemented(*args, **kwargs):
                    raise NotImplementedError(
                        f"Function '{name}' is not available in PyTorch-only mode. "
                        f"The functionality is not yet implemented.")
                return not_implemented
            
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    # Always return the mock module in PyTorch-only mode
    return MockExtModule(funcs)


def check_ops_exist() -> bool:
    """
    In PyTorch-only mode, we return False to indicate that compiled ops don't exist.
    This helps code branches select PyTorch-only implementations.
    """
    return False
