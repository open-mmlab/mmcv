import warnings

import torch

from .builder import RUNNER_BUILDERS, RUNNERS
from mmcv.utils import build_from_cfg

@RUNNER_BUILDERS.register_module()
class DefaultRunnerConstructor:
    """Default constructor for runners.
    """

    def __init__(self, runner_cfg, default_args = None):
        if not isinstance(runner_cfg, dict):
            raise TypeError('runner_cfg should be a dict',
                            f'but got {type(runner_cfg)}')
        self.runner_cfg = runner_cfg
        self.default_args = default_args
        
    def __call__(self):
        return build_from_cfg(self.runner_cfg, RUNNERS, default_args=self.default_args)
