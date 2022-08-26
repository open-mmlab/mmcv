# Copyright (c) OpenMMLab. All rights reserved.
"""This file holding some environment constant for sharing by other files."""

from mmengine.utils.dl_utils import collect_env as mmengine_collect_env

import mmcv


def collect_env():
    """Collect the information of the running environments.

    Returns:
        dict: The environment information. The following fields are contained.

            - sys.platform: The variable of ``sys.platform``.
            - Python: Python version.
            - CUDA available: Bool, indicating if CUDA is available.
            - GPU devices: Device type of each GPU.
            - CUDA_HOME (optional): The env var ``CUDA_HOME``.
            - NVCC (optional): NVCC version.
            - GCC: GCC version, "n/a" if GCC is not installed.
            - MSVC: Microsoft Virtual C++ Compiler version, Windows only.
            - PyTorch: PyTorch version.
            - PyTorch compiling details: The output of \
                ``torch.__config__.show()``.
            - TorchVision (optional): TorchVision version.
            - OpenCV: OpenCV version.
            - MMEngine: MMEngine version.
            - MMCV: MMCV version.
            - MMCV Compiler: The GCC version for compiling MMCV ops.
            - MMCV CUDA Compiler: The CUDA version for compiling MMCV ops.
    """
    env_info = mmengine_collect_env()
    env_info['MMCV'] = mmcv.__version__

    try:
        from mmcv.ops import get_compiler_version, get_compiling_cuda_version
    except ModuleNotFoundError:
        env_info['MMCV Compiler'] = 'n/a'
        env_info['MMCV CUDA Compiler'] = 'n/a'
    else:
        env_info['MMCV Compiler'] = get_compiler_version()
        env_info['MMCV CUDA Compiler'] = get_compiling_cuda_version()

    return env_info
