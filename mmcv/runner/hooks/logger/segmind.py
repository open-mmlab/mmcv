# Copyright (c) OpenMMLab. All rights reserved.
from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class SegmindLoggerHook(LoggerHook):
    """Class to log metrics to Segmind.

    It requires `Segmind`_ to be installed.

    Args:
        interval (int): Logging interval (every k iterations). Default: 10.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default False.
        by_epoch (bool): Whether EpochBasedRunner is used. Default True.

    .. _Segmind:
        https://docs.segmind.com/python-library
    """

    def __init__(self,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True):
        super(SegmindLoggerHook, self).__init__(interval, ignore_last,
                                                reset_flag, by_epoch)
        self.import_segmind()

    def import_segmind(self):
        try:
            import segmind
        except ImportError:
            raise ImportError(
                "Please run 'pip install segmind' to install segmind")
        self.segmind_log_metrics = segmind.tracking.fluent.log_metrics
        self.segmind_mlflow_log = segmind.utils.logging_utils.try_mlflow_log

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if tags:
            # logging metrics to segmind
            self.segmind_mlflow_log(
                self.segmind_log_metrics,
                tags,
                step=runner.epoch,
                epoch=runner.epoch)
