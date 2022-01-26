# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
import warnings

from ..engine import single_gpu_test
from .iter_based_runner import IterBasedRunner
from .epoch_based_runner import EpochBasedRunner
from .builder import RUNNERS
from .hooks import HOOKS
from .hooks.evaluation import EvalHook
from .ipu_utils.util import parse_ipu_options, wrap_model, wrap_data_loader, build_from_cfg_with_wrapper, wrap_lr_update_hook, wrap_optimizer_hook, IPU_MODE

class IpuBaseRunner(metaclass=ABCMeta):
    def __init__(self,ipu_options={},**kwargs):
        super(IpuBaseRunner, self).__init__(**kwargs)
        # process options of ipu
        if IPU_MODE:
            self.ipu_options = parse_ipu_options(ipu_options)
            # self.data_loader = wrap_data_loader(self.data_loader)
            self.model = wrap_model(self.model, self.ipu_options, self.optimizer, self.logger)
            self.ipu_data_loaders_mappin = {} # may have bug in multi-processer
        else:
            warnings.warn('no ipu found, degrade to CPU mode', UserWarning)

    def run(self, data_loaders, *args, **kwargs):
        # map data_loader to ipu data_loader
        if IPU_MODE:
            ipu_data_loaders = []
            for data_loader in data_loaders:
                if data_loader not in self.ipu_data_loaders_mappin:
                    ipu_data_loader = wrap_data_loader(data_loader, self.ipu_options)
                    self.ipu_data_loaders_mappin[data_loader] = ipu_data_loader
                else:
                    ipu_data_loader = self.ipu_data_loaders_mappin[data_loader]
                ipu_data_loaders.append(ipu_data_loader)
                data_loaders = ipu_data_loaders
        super().run(data_loaders, *args, **kwargs)

    def register_lr_hook(self, lr_config):
        if IPU_MODE:
            if lr_config is None:
                raise NotImplementedError
            elif isinstance(lr_config, dict):
                assert 'policy' in lr_config
                policy_type = lr_config.pop('policy')
                # If the type of policy is all in lower case, e.g., 'cyclic',
                # then its first letter will be capitalized, e.g., to be 'Cyclic'.
                # This is for the convenient usage of Lr updater.
                # Since this is not applicable for `
                # CosineAnnealingLrUpdater`,
                # the string will not be changed if it contains capital letters.
                if policy_type == policy_type.lower():
                    policy_type = policy_type.title()
                hook_type = policy_type + 'LrUpdaterHook'
                lr_config['type'] = hook_type
                hook = build_from_cfg_with_wrapper(lr_config, HOOKS, wrap_lr_update_hook)
            else:
                raise NotImplementedError
            self.register_hook(hook, priority='VERY_HIGH')
        else:
            super().register_lr_hook(lr_config)

    def register_optimizer_hook(self, optimizer_config):
        if optimizer_config is None:
            raise RuntimeError('ipu need to wrap optimzier hook, but no optimizer hook set')
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', 'OptimizerHook')
            hook = build_from_cfg_with_wrapper(optimizer_config, HOOKS, wrap_optimizer_hook)
        else:
            raise RuntimeError('ipu need to wrap optimzier hook before inittialization, but seems optimzier hook is initilized')
        self.register_hook(hook, priority='ABOVE_NORMAL')

    def convert_eval_model(self,): 
        # temp ussage to convert eval function used in the runner.evalhook
        # step 1: find the evalhook in runner.hooks
        evalhook = None
        for hook in self.hooks:
            if isinstance (hook, EvalHook):
                evalhook = hook
                break
        assert evalhook is not None, "only .hooks.evaluation.EvalHook is implemented, but not found it"
        # step 2: check the evalhook.test_fn, currently only implemented mmcv.engine.single_gpu_test on IPU
        assert type(evalhook.test_fn) == type(single_gpu_test)
        # step 3: convert evalhook.test_fn
        
                


@RUNNERS.register_module()
class IpuEpochBasedRunner(IpuBaseRunner, EpochBasedRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """
    pass


@RUNNERS.register_module()
class IpuIterBasedRunner(IpuBaseRunner, IterBasedRunner):
    """Iteration-based Runner.

    This runner train models iteration by iteration.
    """
    pass
