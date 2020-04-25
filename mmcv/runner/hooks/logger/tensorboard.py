# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp

import torch

from mmcv.runner import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module
class TensorboardLoggerHook(LoggerHook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True):
        super(TensorboardLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag)
        self.log_dir = log_dir

    @master_only
    def before_run(self, runner):
        if torch.__version__ >= '1.1' and '.' in torch.__version__:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')
        else:
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        for var in runner.log_buffer.output:
            if var in ['time', 'data_time']:
                continue
            tag = f'{var}/{runner.mode}'
            record = runner.log_buffer.output[var]
            if isinstance(record, str):
                self.writer.add_text(tag, record, runner.iter)
            else:
                self.writer.add_scalar(tag, runner.log_buffer.output[var],
                                       runner.iter)
        self.writer.add_scalar('learning_rate',
                               runner.current_lr()[0], runner.iter)
        self.writer.add_scalar('momentum',
                               runner.current_momentum()[0], runner.iter)

    @master_only
    def after_run(self, runner):
        self.writer.close()
