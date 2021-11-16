# Copyright (c) OpenMMLab. All rights reserved.
import os
import shutil
from distutils.dir_util import copy_tree

from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class WandbLoggerHook(LoggerHook):

    def __init__(self,
                 init_kwargs=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 commit=True,
                 by_epoch=True,
                 with_step=True,
                 config_path=None):
        
        super(WandbLoggerHook, self).__init__(interval, ignore_last,
                                              reset_flag, by_epoch)
        
        """
        Args:
            with_step (bool): whether making a log in each step or not.
                Default: True.
            config_path (str, optional): The path of the final config of each
                experiment. It can be either a path of the final config file or
                a directory of config files. The config is uploaded to wandb
                server if it is not None. Default: None.
                `New in version 1.3.18.` 
        """
        
        self.import_wandb()
        self.init_kwargs = init_kwargs
        self.commit = commit
        self.with_step = with_step
        self.config_path = config_path

    def import_wandb(self):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    @master_only
    def before_run(self, runner):
        super(WandbLoggerHook, self).before_run(runner)
        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            self.wandb.init(**self.init_kwargs)
        else:
            self.wandb.init()
        
        if self.config_path is not None:
            if os.path.isdir(self.config_path):
                copy_tree(self.config_path, self.wandb.run.dir)
                for path_under_wandb, _, _ in os.walk(self.wandb.run.dir):
                    self.wandb.save(
                        glob_str=os.path.join(path_under_wandb,'*'),
                        base_path=self.wandb.run.dir,
                        policy='now'
                    )
            else:
                if os.path.isfile(self.config_path):
                    shutil.copy2(self.config_path, self.wandb.run.dir)
                    self.wandb.save(
                        glob_str=os.path.join(self.wandb.run.dir,'*'),
                        base_path=self.wandb.run.dir,
                        policy='now'
                    )
                else:
                    raise FileNotFoundError(
                        "No such file or directory: " + self.config_path
                    )
    
    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if tags:
            if self.with_step:
                self.wandb.log(
                    tags, step=self.get_iter(runner), commit=self.commit)
            else:
                tags['global_step'] = self.get_iter(runner)
                self.wandb.log(tags, commit=self.commit)

    @master_only
    def after_run(self, runner):
        self.wandb.join()
