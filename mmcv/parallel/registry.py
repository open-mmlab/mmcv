from torch.nn.parallel import DataParallel, DistributedDataParallel

from mmcv.utils import Registry

MODULE_WRAPPERS = Registry('module wrapper')
MODULE_WRAPPERS.register_module(DataParallel)
MODULE_WRAPPERS.register_module(DistributedDataParallel)
