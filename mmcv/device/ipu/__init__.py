# Copyright (c) OpenMMLab. All rights reserved.
from .ipu_wrapper import IS_IPU

if IS_IPU:
    from .dataloader import IPUDataLoader
    from .hook_wrapper import IPUFp16OptimizerHook
    from .model_wrapper import ipu_model_wrapper
    from .runner import IPUBaseRunner, IPUEpochBasedRunner, IPUIterBasedRunner
    from .utils import cast_to_options
    __all__ = [
        'cast_to_options', 'ipu_model_wrapper', 'IS_IPU',
        'IPUFp16OptimizerHook', 'IPUDataLoader', 'IPUBaseRunner',
        'IPUEpochBasedRunner', 'IPUIterBasedRunner'
    ]
