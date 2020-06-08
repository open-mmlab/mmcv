# Copyright (c) Open-MMLab. All rights reserved.
from torch.nn.parallel import DataParallel, DistributedDataParallel

from .distributed_deprecated import MMDistributedDataParallel


def is_parallel_module(module):
    """Check if a module is a parallel module.

    The following 3 modules (and their subclasses) are regarded as parallel
    modules: DataParallel, DistributedDataParallel,
    MMDistributedDataParallel (the deprecated version).

    Args:
        module (nn.Module): The module to be checked.

    Returns:
        bool: True if the input module is a parallel module.
    """
    parallels = (DataParallel, DistributedDataParallel,
                 MMDistributedDataParallel)
    if isinstance(module, parallels):
        return True
    else:
        return False
