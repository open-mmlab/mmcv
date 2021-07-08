# Copyright (c) Open-MMLab. All rights reserved.
from torch.nn.parallel import DataParallel, DistributedDataParallel

from mmcv.utils import Registry

MODULE_WRAPPERS = Registry('module wrapper')
MODULE_WRAPPERS.register_module(module=DataParallel)
MODULE_WRAPPERS.register_module(module=DistributedDataParallel)
