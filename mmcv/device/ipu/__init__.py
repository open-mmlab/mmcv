# Copyright (c) OpenMMLab. All rights reserved.
from .utils import IS_IPU


if IS_IPU:
    from .model_wrapper import (cast_to_options, ipu_model_wrapper,
                                build_from_cfg_with_wrapper, model_sharding,
                                recomputation_checkpoint)
    from .hook_wrapper import (wrap_optimizer_hook, IPUFp16OptimizerHook,
                               wrap_lr_updater_hook)
    from .dataloader import IPUDataLoader
    from .runner import IPUBaseRunner, IPUEpochBasedRunner, IPUIterBasedRunner
    __all__ = [
        'cast_to_options', 'ipu_model_wrapper',
        'build_from_cfg_with_wrapper', 'IS_IPU',
        'model_sharding', 'wrap_optimizer_hook',
        'IPUFp16OptimizerHook', 'wrap_lr_updater_hook',
        'recomputation_checkpoint', 'IPUDataLoader',
        'IPUBaseRunner', 'IPUEpochBasedRunner',
        'IPUIterBasedRunner'
    ]
