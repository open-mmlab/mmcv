# Copyright (c) OpenMMLab. All rights reserved.
from .flops_counter import get_model_complexity_info
from .fuse_conv_bn import fuse_conv_bn

__all__ = ['get_model_complexity_info', 'fuse_conv_bn']
