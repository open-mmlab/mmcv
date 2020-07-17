# Copyright (c) Open-MMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from ..hook import Hook
from ...dist_utils import get_dist_info
import torch.distributed as dist
import torch


class LoggerHook(Hook):
    """Base class for logger hooks.

    Args:
        interval (int): Logging interval (every k iterations).
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
        reset_flag (bool): Whether to clear the output buffer after logging.
        by_epoch (bool): Whether EpochBasedRunner is used.
    """

    __metaclass__ = ABCMeta

    def __init__(self,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True):
        self.interval = interval
        self.ignore_last = ignore_last
        self.reset_flag = reset_flag
<<<<<<< HEAD
        _rank, _world_size = get_dist_info()
        self.rank = _rank
        self.world_size = _world_size
=======
        self.by_epoch = by_epoch
>>>>>>> upstream/master

    @abstractmethod
    def log(self, runner):
        pass

    def before_run(self, runner):
        for hook in runner.hooks[::-1]:
            if isinstance(hook, LoggerHook):
                hook.reset_flag = True
                break

    def before_epoch(self, runner):
        runner.log_buffer.clear()  # clear logs of last epoch

    def after_train_iter(self, runner):
        if self.by_epoch and self.every_n_inner_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
        elif not self.by_epoch and self.every_n_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
            self.sync_buffer_output(runner)
        elif self.end_of_epoch(runner) and not self.ignore_last:
            runner.log_buffer.average(self.interval)
            self.sync_buffer_output(runner)
        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def after_val_iter(self, runner):
        if self.every_n_inner_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
            self.sync_buffer_output(runner)
        elif self.end_of_epoch(runner) and not self.ignore_last:
            runner.log_buffer.average(self.interval)
            self.sync_buffer_output(runner)
        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def after_train_epoch(self, runner):
        runner.log_buffer.average()
        self.sync_buffer_output(runner)
        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def after_val_epoch(self, runner):
        runner.log_buffer.average()
        self.sync_buffer_output(runner)
        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def sync_buffer_output(self, runner):
        for k, v in runner.log_buffer.output.items():
            tmp_tensor = torch.Tensor([v]).cuda(torch.cuda.current_device())
            dist.all_reduce(tmp_tensor)
            tmp_tensor.div_(self.world_size)
            runner.log_buffer.output[k] = tmp_tensor.item()
