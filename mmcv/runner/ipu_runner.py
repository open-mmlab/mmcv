# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta

from .iter_based_runner import IterBasedRunner
from .epoch_based_runner import EpochBasedRunner
from .builder import RUNNERS
from .hooks import HOOKS, LrUpdaterHook

from mmcv.runner.ipu import IPU_MODE
if IPU_MODE:
    from mmcv.runner.ipu import parse_ipu_options, build_from_cfg_with_wrapper,\
        IPU_MODE, ipu_model_wrapper, wrap_optimizer_hook,\
        IpuFp16OptimizerHook


def wrap_lr_update_hook(lr_hook_class,):
    assert issubclass(lr_hook_class, LrUpdaterHook)

    class ipu_lr_hook_class(lr_hook_class):
        def _set_lr(self, runner, *args, **kwargs):
            result = super()._set_lr(runner, *args, **kwargs)
            assert result is None  # _set_lr should return nothing
            runner.model.setOptimizer(runner.optimizer)
    return ipu_lr_hook_class


class IpuBaseRunner(metaclass=ABCMeta):
    def __init__(
            self,
            ipu_options={},
            modules_to_record=[],
            pipeline_cfg={},
            fp16_cfg=None,
            **kwargs):
        super(IpuBaseRunner, self).__init__(**kwargs)
        # process options of ipu
        if IPU_MODE:
            self.ipu_options = parse_ipu_options(ipu_options)
            # self.data_loader = wrap_data_loader(self.data_loader)
            self.model = ipu_model_wrapper(
                self.model, self.ipu_options, self.optimizer, self.logger,
                modules_to_record=modules_to_record, pipeline_cfg=pipeline_cfg,
                fp16_cfg=fp16_cfg)
        else:
            # warnings.warn('no ipu found, degrade to CPU mode', UserWarning)
            raise NotImplementedError('cpu mode on IpuRunner still has bug')

    def register_lr_hook(self, lr_config):
        if IPU_MODE:
            if lr_config is None:
                raise NotImplementedError
            elif isinstance(lr_config, dict):
                assert 'policy' in lr_config
                policy_type = lr_config.pop('policy')
                # If the type of policy is all in lower case,
                # e.g., 'cyclic', then its first letter will be capitalized,
                # e.g., to be 'Cyclic'.
                # This is for the convenient usage of Lr updater.
                # Since this is not applicable for `
                # CosineAnnealingLrUpdater`, the string will not be changed
                # if it contains capital letters.
                if policy_type == policy_type.lower():
                    policy_type = policy_type.title()
                hook_type = policy_type + 'LrUpdaterHook'
                lr_config['type'] = hook_type
                hook = build_from_cfg_with_wrapper(
                    lr_config, HOOKS, wrap_lr_update_hook)
            else:
                raise NotImplementedError
            self.register_hook(hook, priority='VERY_HIGH')
        else:
            super().register_lr_hook(lr_config)

    def register_optimizer_hook(self, optimizer_config):
        if optimizer_config is None:
            raise RuntimeError(
                'ipu need to wrap optimzier hook, but no optimizer hook set')
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', 'OptimizerHook')
            hook = build_from_cfg_with_wrapper(
                optimizer_config, HOOKS, wrap_optimizer_hook)
        elif isinstance(optimizer_config, IpuFp16OptimizerHook):
            hook = optimizer_config
        else:
            raise RuntimeError(
                'ipu need to wrap optimzier hook before inittialization,\
                    but seems optimzier hook is initilized')
        self.register_hook(hook, priority='ABOVE_NORMAL')


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
