# flake8: noqa
# Copyright (c) Open-MMLab. All rights reserved.
from .config import Config, ConfigDict, DictAction
from .misc import (check_prerequisites, concat_list, deprecated_api_warning,
                   is_list_of, is_seq_of, is_str, is_tuple_of, iter_cast,
                   list_cast, requires_executable, requires_package,
                   slice_list, tuple_cast)
from .path import (check_file_exist, fopen, is_filepath, mkdir_or_exist,
                   scandir, symlink)
from .progressbar import (ProgressBar, track_iter_progress,
                          track_parallel_progress, track_progress)
from .timer import Timer, TimerError, check_time
from .version_utils import get_git_hash

try:
    import torch
except ImportError:
    __all__ = [
        'Config', 'ConfigDict', 'DictAction', 'is_str', 'iter_cast',
        'list_cast', 'tuple_cast', 'is_seq_of', 'is_list_of', 'is_tuple_of',
        'slice_list', 'concat_list', 'check_prerequisites', 'requires_package',
        'requires_executable', 'is_filepath', 'fopen', 'check_file_exist',
        'mkdir_or_exist', 'symlink', 'scandir', 'ProgressBar',
        'track_progress', 'track_iter_progress', 'track_parallel_progress',
        'Timer', 'TimerError', 'check_time', 'deprecated_api_warning',
        'get_git_hash'
    ]
else:
    from .env import TORCH_VERSION
    from .logging import get_logger, print_log
    from .parrots_wrapper import (CUDA_HOME, BuildExtension, CppExtension,
                                  CUDAExtension, DataLoader, PoolDataLoader,
                                  SyncBatchNorm, _AdaptiveAvgPoolNd,
                                  _AdaptiveMaxPoolNd, _AvgPoolNd, _BatchNorm,
                                  _ConvNd, _ConvTransposeMixin, _InstanceNorm,
                                  _MaxPoolNd, get_build_config)
    from .registry import Registry, build_from_cfg
    __all__ = [
        'Config', 'ConfigDict', 'DictAction', 'get_logger', 'print_log',
        'is_str', 'iter_cast', 'list_cast', 'tuple_cast', 'is_seq_of',
        'is_list_of', 'is_tuple_of', 'slice_list', 'concat_list',
        'check_prerequisites', 'requires_package', 'requires_executable',
        'is_filepath', 'fopen', 'check_file_exist', 'mkdir_or_exist',
        'symlink', 'scandir', 'ProgressBar', 'track_progress',
        'track_iter_progress', 'track_parallel_progress', 'Registry',
        'build_from_cfg', 'Timer', 'TimerError', 'check_time', 'CUDA_HOME',
        'SyncBatchNorm', '_AdaptiveAvgPoolNd', '_AdaptiveMaxPoolNd',
        '_AvgPoolNd', '_BatchNorm', '_ConvNd', '_ConvTransposeMixin',
        '_InstanceNorm', '_MaxPoolNd', 'get_build_config', 'BuildExtension',
        'CppExtension', 'CUDAExtension', 'DataLoader', 'PoolDataLoader',
        'TORCH_VERSION', 'deprecated_api_warning', 'get_git_hash'
    ]
