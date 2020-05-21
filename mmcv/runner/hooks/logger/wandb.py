# Copyright (c) Open-MMLab. All rights reserved.
import numbers

from mmcv.runner import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class WandbLoggerHook(LoggerHook):

    def __init__(self,
                 init_kwargs=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True):
        super(WandbLoggerHook, self).__init__(interval, ignore_last,
                                              reset_flag)
        self.import_wandb()
        self.init_kwargs = init_kwargs

    def import_wandb(self):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    @master_only
    def before_run(self, runner):
        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            self.wandb.init(**self.init_kwargs)
        else:
            self.wandb.init()

    @master_only
    def log(self, runner):
        metrics = {}
        for var, val in runner.log_buffer.output.items():
            if var in ['time', 'data_time']:
                continue
            tag = f'{var}/{runner.mode}'
            if isinstance(val, numbers.Number):
                metrics[tag] = val
        metrics['learning_rate'] = runner.current_lr()[0]
        metrics['momentum'] = runner.current_momentum()[0]
        if metrics:
            self.wandb.log(metrics, step=runner.iter)

    @master_only
    def after_run(self, runner):
        self.wandb.join()
