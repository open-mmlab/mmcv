# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta

from .iter_based_runner import IterBasedRunner
from .epoch_based_runner import EpochBasedRunner
from .builder import RUNNERS
from .hooks import HOOKS

from mmcv.runner.ipu import IPU_MODE
if IPU_MODE:
    from mmcv.runner.ipu import parse_ipu_options,\
        build_from_cfg_with_wrapper, IPU_MODE,\
        ipu_model_wrapper, wrap_optimizer_hook,\
        IpuFp16OptimizerHook, wrap_lr_update_hook


class IpuBaseRunner(metaclass=ABCMeta):
    def __init__(
            self,
            ipu_options={},
            modules_to_record=[],
            ipu_model_cfg={},
            fp16_cfg=None,
            **kwargs):
        super(IpuBaseRunner, self).__init__(**kwargs)
        # process options of ipu
        if IPU_MODE:
            self.ipu_options = parse_ipu_options(ipu_options)
            # self.data_loader = wrap_data_loader(self.data_loader)
            self.model = ipu_model_wrapper(
                self.model, self.ipu_options, self.optimizer, self.logger,
                modules_to_record=modules_to_record,
                ipu_model_cfg=ipu_model_cfg, fp16_cfg=fp16_cfg)
        else:
            # warnings.warn('no ipu found, degrade to CPU mode', UserWarning)
            raise NotImplementedError('cpu mode on IpuRunner not supported')

    def register_lr_hook(self, lr_config):
        assert isinstance(lr_config, dict)
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
        self.register_hook(hook, priority='VERY_HIGH')

    def register_optimizer_hook(self, optimizer_config):
        assert isinstance(optimizer_config, (dict, IpuFp16OptimizerHook))
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', 'OptimizerHook')
            hook = build_from_cfg_with_wrapper(
                optimizer_config, HOOKS, wrap_optimizer_hook)
        else:
            hook = optimizer_config
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
