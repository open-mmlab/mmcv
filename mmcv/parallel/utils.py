# Copyright (c) OpenMMLab. All rights reserved.
from .registry import MODULE_WRAPPERS


def is_module_wrapper(module):
    """Check if a module is a module wrapper.

    The following 3 modules in MMCV (and their subclasses) are regarded as
    module wrappers: DataParallel, DistributedDataParallel,
    MMDistributedDataParallel (the deprecated version). You may add you own
    module wrapper by registering it to mmcv.parallel.MODULE_WRAPPERS or
    its child registry.

    Args:
        module (nn.Module): The module to be checked.

    Returns:
        bool: True if the input module is a module wrapper.
    """

    def dfs(MODULE_WRAPPER):
        module_wrappers = tuple(MODULE_WRAPPER.module_dict.values())
        if isinstance(module, module_wrappers):
            return True
        for CHILD_MODULE_WRAPPER in MODULE_WRAPPER.children.values():
            if dfs(CHILD_MODULE_WRAPPER):
                return True
        return False

    return dfs(MODULE_WRAPPERS)
