# Copyright (c) Open-MMLab. All rights reserved.
from .hook import HOOKS, Hook


@HOOKS.register_module()
class DistSamplerSeedHook(Hook):

    def before_epoch(self, runner):
        if hasattr(runner.data_loader.sampler, 'set_epoch'):
            # in case the data loader uses `SequentialSampler` in Pytorch
            runner.data_loader.sampler.set_epoch(runner.epoch)
        elif hasattr(runner.data_loader.batch_sampler.sampler, 'set_epoch'):
            # batch sampler in pytorch warps the sampler as its attributes.
            runner.data_loader.batch_sampler.sampler.set_epoch(runner.epoch)
