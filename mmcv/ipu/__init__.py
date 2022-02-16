# Copyright (c) OpenMMLab. All rights reserved.
try:
    import poptorch
    IPU_MODE = True
except ImportError:
    IPU_MODE = False

if IPU_MODE:
    from .util import parse_ipu_options, ipu_model_wrapper, build_from_cfg_with_wrapper, add_split_edges
    from .optimizer_hooks import wrap_optimizer_hook, IpuFp16OptimizerHook
    __all__ = [
        'parse_ipu_options', 'ipu_model_wrapper', 'build_from_cfg_with_wrapper', 'IPU_MODE', 'add_split_edges',
        'wrap_optimizer_hook', 'IpuFp16OptimizerHook'
    ]
