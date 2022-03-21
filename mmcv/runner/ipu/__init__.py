# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils.ipu_wrapper import IPU_MODE


if IPU_MODE:
    from .util import parse_ipu_options, ipu_model_wrapper,\
        build_from_cfg_with_wrapper, model_sharding,\
        recomputation_checkpoint
    from .hooks import wrap_optimizer_hook, IPUFp16OptimizerHook,\
        wrap_lr_update_hook
    from .dataloder import IPUDataloader
    __all__ = [
        'parse_ipu_options', 'ipu_model_wrapper',
        'build_from_cfg_with_wrapper', 'IPU_MODE',
        'model_sharding', 'wrap_optimizer_hook',
        'IPUFp16OptimizerHook', 'wrap_lr_update_hook',
        'recomputation_checkpoint', 'IPUDataloader'
    ]
