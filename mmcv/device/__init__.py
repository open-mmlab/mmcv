# Copyright (c) OpenMMLab. All rights reserved.
from . import ipu, mlu, mps, npu, xpu
from .scatter_gather import scatter, scatter_kwargs
from .utils import get_device

__all__ = [
    'npu', 'mlu', 'ipu', 'mps', 'xpu', 'get_device', 'scatter',
    'scatter_kwargs'
]
