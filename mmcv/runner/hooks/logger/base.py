# Copyright (c) Open-MMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from ..hook import Hook


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
        self.by_epoch = by_epoch

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
        elif self.end_of_epoch(runner) and not self.ignore_last:
            # not precise but more stable
            runner.log_buffer.average(self.interval)

        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def after_train_epoch(self, runner):
        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def after_val_epoch(self, runner):
        runner.log_buffer.average()
        self.log(runner)
        if self.reset_flag:
            runner.log_buffer.clear_output()
