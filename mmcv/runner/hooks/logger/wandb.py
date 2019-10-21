from ...utils import master_only
from .base import LoggerHook

import numbers
import wandb


class WandbLoggerHook(LoggerHook):
    def __init__(self, interval=10, ignore_last=True, reset_flag=False):
        super().__init__(WandbLoggerHook, self).__init__(interval, ignore_last, reset_flag)
        self.interval = interval
        self.ignore_last = ignore_last
        self.reset_flag = reset_flag

        wandb.init()  # need to initialize wandb before run

    @master_only
    def log(self, runner):
        metrics = {}
        for var, val in runner.log_buffer.output.items():
            if var in ['time', 'data_time']:
                continue
            tag = '{}/{}'.format(var, runner.mode)
            runner.log_buffer.output[var]
            if isinstance(val, numbers.Number):
                metrics[tag] = val
        wandb.log(metrics, step=runner.iter)

    @master_only
    def after_run(self, runner):
        wandb.join()
