# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn.utils.flops_counter import get_model_complexity_info
from mmcv.cnn.utils.fuse_conv_bn import fuse_conv_bn

__all__ = ['fuse_conv_bn', 'get_model_complexity_info']
