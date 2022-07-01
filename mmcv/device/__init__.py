# Copyright (c) OpenMMLab. All rights reserved.
from . import ipu, mlu, mps
from .scatter_gather import scatter, scatter_kwargs
from .utils import get_device

__all__ = ['mlu', 'ipu', 'mps', 'get_device', 'scatter', 'scatter_kwargs']
