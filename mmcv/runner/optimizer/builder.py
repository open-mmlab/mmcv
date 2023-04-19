# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
from typing import Dict, List

import torch

from mmcv.utils import IS_NPU_AVAILABLE, Registry, build_from_cfg

OPTIMIZERS = Registry('optimizer')
OPTIMIZER_BUILDERS = Registry('optimizer builder')


def register_torch_optimizers() -> List:
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            OPTIMIZERS.register_module()(_optim)
            torch_optimizers.append(module_name)
    if IS_NPU_AVAILABLE:
        torch_npu_optimizers = register_torch_npu_optimizers(torch_optimizers)
        torch_optimizers.extend(torch_npu_optimizers)
    return torch_optimizers


def register_torch_npu_optimizers(torch_optimizers) -> List:

    import torch_npu
    if not hasattr(torch_npu, 'optim'):
        return []

    torch_npu_optimizers = []
    for module_name in dir(torch_npu.optim):
        if module_name.startswith('__') or module_name in torch_optimizers:
            continue
        _optim = getattr(torch_npu.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            OPTIMIZERS.register_module()(_optim)
            torch_npu_optimizers.append(module_name)
    return torch_npu_optimizers


TORCH_OPTIMIZERS = register_torch_optimizers()


def build_optimizer_constructor(cfg: Dict):
    return build_from_cfg(cfg, OPTIMIZER_BUILDERS)


def build_optimizer(model, cfg: Dict):
    optimizer_cfg = copy.deepcopy(cfg)
    constructor_type = optimizer_cfg.pop('constructor',
                                         'DefaultOptimizerConstructor')
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    optim_constructor = build_optimizer_constructor(
        dict(
            type=constructor_type,
            optimizer_cfg=optimizer_cfg,
            paramwise_cfg=paramwise_cfg))
    optimizer = optim_constructor(model)
    return optimizer
