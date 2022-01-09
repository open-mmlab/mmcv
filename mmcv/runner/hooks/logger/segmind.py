# Copyright (c) OpenMMLab. All rights reserved.
import numbers

from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class SegmindLoggerHook(LoggerHook):

    def __init__(self,
                 init_kwargs=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True):
        super(SegmindLoggerHook, self).__init__(interval, ignore_last,
                                                reset_flag)
        self.import_segmind()

    def import_segmind(self):
        try:
            import segmind
            from segmind.tracking.fluent import log_metrics
            from segmind.utils.logging_utils import try_mlflow_log
        except ImportError:
            raise ImportError("Please run 'pip install segmind' to install \
                segmind")
        self.segmind = segmind
        self.log_metrics = log_metrics
        self.try_mlflow_log = try_mlflow_log

    @master_only
    def before_run(self, runner):
        super(SegmindLoggerHook, self).before_run(runner)
        if self.segmind is None:
            self.import_segmind()

    @master_only
    def log(self, runner):
        metrics = {}

        for var, val in runner.log_buffer.output.items():
            if var in ['time', 'data_time']:
                continue

            tag = f'{var}_{runner.mode}'
            if isinstance(val, numbers.Number):
                metrics[tag] = val

        metrics['learning_rate'] = runner.current_lr()[0]
        metrics['momentum'] = runner.current_momentum()[0]

        # logging metrics to segmind
        self.try_mlflow_log(
            self.log_metrics, metrics, step=runner.epoch, epoch=runner.epoch)

    @master_only
    def after_run(self, runner):
        pass
