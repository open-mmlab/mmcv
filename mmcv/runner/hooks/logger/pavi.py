# Copyright (c) Open-MMLab. All rights reserved.
import numbers
import os.path as osp

import numpy as np
import torch

from mmcv.runner import master_only
from ..hook import HOOKS
from .base import LoggerHook


def is_scalar(val, include_np=True, include_torch=True):
    """Tell the input variable is a scalar or not.

    Args:
        val: Input variable.
        include_np (bool): Whether include 0-d np.ndarray as a scalar.
        include_torch (bool): Whether include 0-d torch.Tensor as a scalar.

    Returns:
        bool: True or False.
    """
    if isinstance(val, numbers.Number):
        return True
    elif include_np and isinstance(val, np.ndarray) and val.ndim == 0:
        return True
    elif include_torch and isinstance(val, torch.Tensor) and len(val) == 1:
        return True
    else:
        return False


@HOOKS.register_module()
class PaviLoggerHook(LoggerHook):

    def __init__(self,
                 init_kwargs=None,
                 add_graph=False,
                 add_last_ckpt=False,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True):
        super(PaviLoggerHook, self).__init__(interval, ignore_last, reset_flag)
        self.init_kwargs = init_kwargs
        self.add_graph = add_graph
        self.add_last_ckpt = add_last_ckpt

    @master_only
    def before_run(self, runner):
        try:
            from pavi import SummaryWriter
        except ImportError:
            raise ImportError('Please run "pip install pavi" to install pavi.')

        self.run_name = runner.work_dir.split('/')[-1]

        if not self.init_kwargs:
            self.init_kwargs = dict()
        self.init_kwargs['task'] = self.run_name
        self.init_kwargs['model'] = runner._model_name

        self.writer = SummaryWriter(**self.init_kwargs)

        if self.add_graph:
            self.writer.add_graph(runner.model)

    @master_only
    def log(self, runner):
        tags = {}
        for tag, val in runner.log_buffer.output.items():
            if tag not in ['time', 'data_time'] and is_scalar(val):
                tags[tag] = val
        tags['learning_rate'] = runner.current_lr()[0]
        tags['momentum'] = runner.current_momentum()[0]
        if tags:
            self.writer.add_scalars(runner.mode, tags, runner.iter)

    @master_only
    def after_run(self, runner):
        if self.add_last_ckpt:
            ckpt_path = osp.join(runner.work_dir, 'latest.pth')
            self.writer.add_snapshot_file(
                tag=self.run_name,
                snapshot_file_path=ckpt_path,
                iteration=runner.iter)
