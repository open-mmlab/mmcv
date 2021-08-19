# flake8: noqa
# Copyright (c) OpenMMLab. All rights reserved.
from .builder import PIPELINES
from .pipelines.transforms import Normalize

__all__ = ['PIPELINES', 'Normalize']
try:
    import torch
except ImportError:
    pass
else:
    from .pipelines.formatting import ImageToTensor, ToTensor
    __all__ += ['ImageToTensor', 'ToTensor']
