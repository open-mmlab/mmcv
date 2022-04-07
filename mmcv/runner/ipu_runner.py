# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .base_runner import BaseRunner
from .iter_based_runner import IterBasedRunner
from .epoch_based_runner import EpochBasedRunner
from .builder import RUNNERS
from .hooks import HOOKS

from mmcv.runner.ipu import IPU_MODE
if IPU_MODE:
    from mmcv.runner.ipu import (parse_ipu_options,
                                 build_from_cfg_with_wrapper, IPU_MODE,
                                 ipu_model_wrapper, wrap_optimizer_hook,
                                 IPUFp16OptimizerHook, wrap_lr_update_hook,
                                 IPUDataloader)


class IPUBaseRunner(BaseRunner):
    """A base runner for IPU.

    This runner has some extra processes for IPU which are shown below:

    1. Parse options for IPU
    2. wrap pytorch model for IPU
    3. Raise errors while encountering illegal usage
    4. Input IPU options and initialize dataloader if finding an instance of
       of IPUDataloader

    Args:
        model (:obj:`nn.Module`): The model to be run.
        options (mmcv.Config, dict): Options that will be used to compile
            and run the model.
        modules_to_record (mmcv.Config, list): Index or name of modules which
            will be recorded for output. It is necessary to specify output for
            static graph of model training or inference.
        ipu_model_cfg (mmcv.Config, dict): Config of model partition and
            recomputing checkpoint
        fp16_cfg (mmcv.Config): Config for fp16 training.
        batch_processor (callable): A callable method that process a data
            batch. Should be None for IPU runner
        kwargs (Dict[str, Any], optional): Keyword arguments will be passed to ``base_runner.BaseRunner``.
    """
    def __init__(
            self,
            model,
            ipu_options=None,
            modules_to_record=None,
            ipu_model_cfg=None,
            fp16_cfg=None,
            batch_processor=None,
            **kwargs):
        assert hasattr(model, 'train_step') and batch_processor is None,\
            'only support model with train_step'

        if ipu_options is None:
            ipu_options = {}
        # call BaseRunner.__init__() here
        super().__init__(model, **kwargs)

        # process options of ipu
        if IPU_MODE:
            self.ipu_options = parse_ipu_options(ipu_options)
            self.model = ipu_model_wrapper(
                self.model, self.ipu_options, self.optimizer, self.logger,
                modules_to_record=modules_to_record,
                ipu_model_cfg=ipu_model_cfg, fp16_cfg=fp16_cfg)
        else:
            raise NotImplementedError('cpu mode on IPURunner not supported')

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
        assert isinstance(optimizer_config, (dict, IPUFp16OptimizerHook))
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', 'OptimizerHook')
            hook = build_from_cfg_with_wrapper(
                optimizer_config, HOOKS, wrap_optimizer_hook)
        else:
            hook = optimizer_config
        self.register_hook(hook, priority='ABOVE_NORMAL')

    def run(self, data_loaders, workflow, *args, **kwargs):
        for i, flow in enumerate(workflow):
            mode, _ = flow
            # initialize IPU dataloader if not initialized
            assert isinstance(data_loaders[i], IPUDataloader),\
                'IPU runner can only work with `IPUDataloader`'
            data_loaders[i].init(options=self.get_ipu_options(mode))

        super().run(data_loaders, workflow, *args, **kwargs)

    def get_ipu_options(self, mode):
        if mode == 'train':
            return self.ipu_options['training']
        elif mode == 'val':
            return self.ipu_options['inference']
        else:
            raise ValueError(f'mode should be train or val but got {mode}')


@RUNNERS.register_module()
class IPUEpochBasedRunner(IPUBaseRunner, EpochBasedRunner):
    """Epoch-based Runner for IPU.

    The Inheritance order(MRO) is:
    IPUEpochBasedRunner -> IPUBaseRunner -> EpochBasedRunner -> BaseRunner
    This runner train models epoch by epoch.
    """
    pass


@RUNNERS.register_module()
class IPUIterBasedRunner(IPUBaseRunner, IterBasedRunner):
    """Iteration-based Runner for IPU.

    The Inheritance order(MRO) is:
    IPUIterBasedRunner -> IPUBaseRunner -> IterBasedRunner -> BaseRunner
    This runner train models iteration by iteration.
    """
    pass
