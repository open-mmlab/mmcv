# Copyright (c) OpenMMLab. All rights reserved.


def is_ipex_avaiable() -> bool:
    """Just check whether intel extension for pytorch is installed.

    ipex can also optimize models for Intel CPU only and doesn't need an XPU,
    in such case, users can also use XPUBaseRunner to optimize models for CPU.
    """
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401

        return True
    except ImportError:
        return False


if is_ipex_avaiable():
    from .runner import XPUBaseRunner, XPUEpochBasedRunner, XPUIterBasedRunner

    __all__ = ['XPUBaseRunner', 'XPUEpochBasedRunner', 'XPUIterBasedRunner']
