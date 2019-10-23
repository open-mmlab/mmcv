from ...utils import master_only
from .base import LoggerHook

import numbers
import wandb


class WandbLoggerHook(LoggerHook):
    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True):
        super(WandbLoggerHook, self).__init__(interval, ignore_last,
                                              reset_flag)
        self.log_dir = log_dir

        self.init_wandb()

    @master_only
    def init_wandb(self):
        wandb.init()

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
