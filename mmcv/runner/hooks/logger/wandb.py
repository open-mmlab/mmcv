# Copyright (c) Open-MMLab. All rights reserved.
from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class WandbLoggerHook(LoggerHook):

    def __init__(self,
                 init_kwargs=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 commit=True,
                 by_epoch=True):
        super(WandbLoggerHook, self).__init__(interval, ignore_last,
                                              reset_flag, by_epoch)
        self.import_wandb()
        self.init_kwargs = init_kwargs
        self.commit = commit

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
        tags = self.get_loggable_tags(runner)
        if tags:
            self.wandb.log(
                tags, step=self.get_iter(runner), commit=self.commit)

    @master_only
    def after_run(self, runner):
        self.wandb.join()
