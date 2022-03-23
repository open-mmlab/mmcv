# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
import torch

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


class IPUBaseRunner(metaclass=ABCMeta):
    """ A base runner for IPU device, should be inherited with
    base_runner.BaseRunner
    This runner has some extra processes for IPU which are showed below:
    1. Parse options for IPU
    2. wrap pytorch model for IPU
    3. Raise errors while encountering illegal ussage
    4. Input IPU options and initialize dataloader if finding a instance of
       of IPUDataloader
    Args:
        model (pytorch.model)
        options (mmcv.Config, dict): Options that will be used to compile
            and run the model.
        modules_to_record (mmcv.Config, list): idx or name of modules which
            will be recorded for output. It is necessary to specify output for
            static graph of model training or inference.
        ipu_model_cfg (mmcv.Config, dict): config of model partition and
            recomputing checkpoint
        fp16_cfg (mmcv.Config): config for fp16 training
        batch_processor (callable): A callable method that process a data
            batch. Should be None for IPU runner
        kwargs: other kwargs please check Class base_runner.BaseRunner
    """
    def __init__(
            self,
            model,
            ipu_options={},
            modules_to_record=[],
            ipu_model_cfg={},
            fp16_cfg=None,
            ipu_dataloader=False,
            batch_processor=None,
            **kwargs):
        assert hasattr(model, 'train_step') and batch_processor is None,\
            'only support model with train_step'
        if isinstance(model, torch.nn.parallel.DataParallel):
            raise TypeError(
                'if you want to implement data parallelism '
                'at the module level on IPU, '
                'use IPU option: replicationFactor')

        super(IPUBaseRunner, self).__init__(model, **kwargs)

        # process options of ipu
        if IPU_MODE:
            self.ipu_options = parse_ipu_options(ipu_options)
            # self.data_loader = wrap_data_loader(self.data_loader)
            self.model = ipu_model_wrapper(
                self.model, self.ipu_options, self.optimizer, self.logger,
                modules_to_record=modules_to_record,
                ipu_model_cfg=ipu_model_cfg, fp16_cfg=fp16_cfg)
            self.ipu_dataloader = ipu_dataloader
        else:
            # warnings.warn('no ipu found, degrade to CPU mode', UserWarning)
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

    def run(self, data_loaders, *args, **kwargs):
        # map data_loader to ipu data_loader
        if self.ipu_dataloader:
            training_opts = self.ipu_options['training']

            for data_loader in data_loaders:
                if isinstance(data_loader, IPUDataloader)\
                   and not data_loader.initialized:
                    data_loader.init(options=training_opts)

        super().run(data_loaders, *args, **kwargs)


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
