import sys

import pytest

import mmcv
from mmcv.utils import collect_env


def test_collect_env():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip('skipping tests that require PyTorch')

    env_info = collect_env()
    expected_keys = [
        'sys.platform', 'Python', 'CUDA available', 'CUDA_HOME', 'NVCC',
        'PyTorch', 'PyTorch compiling details', 'OpenCV', 'MMCV',
        'MMCV Compiler', 'MMCV CUDA Compiler'
    ]
    for key in expected_keys:
        assert key in env_info

    if sys.platform != 'win32':
        assert 'GCC' in env_info

    assert env_info['sys.platform'] == sys.platform
    assert env_info['Python'] == sys.version.replace('\n', '')
    assert env_info['MMCV'] == mmcv.__version__
