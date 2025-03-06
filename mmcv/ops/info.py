# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from torch import nn
import torch



# PyTorch-only implementation
class InfoModule:
    @staticmethod
    def get_compiler_version(*args, **kwargs):
        warnings.warn("Using PyTorch-only implementation of get_compiler_version. "
                     "This may not be as efficient as the CUDA version.", stacklevel=2)
        
        # For output tensors, zero them out
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                arg.zero_()
        return
    @staticmethod
    def get_compiling_cuda_version(*args, **kwargs):
        warnings.warn("Using PyTorch-only implementation of get_compiling_cuda_version. "
                     "This may not be as efficient as the CUDA version.", stacklevel=2)
        
        # For output tensors, zero them out
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                arg.zero_()
        return

# Create a module-like object to replace ext_module
ext_module = InfoModule


def get_compiler_version():
    return ext_module.get_compiler_version()

def get_compiling_cuda_version():
    return ext_module.get_compiling_cuda_version()
