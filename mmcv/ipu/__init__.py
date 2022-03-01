# Copyright (c) OpenMMLab. All rights reserved.
try:
    import poptorch
    IPU_MODE = True
except ImportError:
    IPU_MODE = False

if IPU_MODE:
    from .util import parse_ipu_options, ipu_model_wrapper,\
        build_from_cfg_with_wrapper, model_sharding
    from .optimizer_hooks import wrap_optimizer_hook, IpuFp16OptimizerHook
    __all__ = [
        'parse_ipu_options', 'ipu_model_wrapper',
        'build_from_cfg_with_wrapper', 'IPU_MODE',
        'model_sharding', 'wrap_optimizer_hook',
        'IpuFp16OptimizerHook'
    ]
