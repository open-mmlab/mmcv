# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import IS_IPU_AVAILABLE
from .static_func import IPUIdentity, slice_statically

if IS_IPU_AVAILABLE:
    from .dataloader import IPUDataLoader
    from .hook_wrapper import IPUFp16OptimizerHook
    from .ipu_hooks import BNToFP32
    from .model_wrapper import build_ipu_model, ipu_model_wrapper
    from .runner import IPUBaseRunner, IPUEpochBasedRunner, IPUIterBasedRunner
    from .utils import cfg2options
    __all__ = [
        'cfg2options', 'ipu_model_wrapper', 'IPUFp16OptimizerHook',
        'IPUDataLoader', 'IPUBaseRunner', 'IPUEpochBasedRunner',
        'IPUIterBasedRunner', 'BNToFP32', 'build_ipu_model',
        'slice_statically', 'IPUIdentity'
    ]
