# flake8: noqa
# Copyright (c) OpenMMLab. All rights reserved.
from .transforms import Normalize

__all__ = ['Normalize']

try:
    import torch
except ImportError:
    pass
else:
    from .formatting import ImageToTensor, ToTensor
    __all__ += ['ImageToTensor', 'ToTensor']
