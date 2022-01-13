# Copyright (c) OpenMMLab. All rights reserved.
from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class SegmindLoggerHook(LoggerHook):
    """Class to log metrics to Segmind.

    It requires `Segmind`_ to be installed.

    .. _Segmind:
        https://docs.segmind.com/python-library
    """

    def __init__(self):
        super(SegmindLoggerHook, self).__init__()
        self.import_segmind()

    def import_segmind(self):
        try:
            import segmind
            from segmind.tracking.fluent import log_metrics
            from segmind.utils.logging_utils import try_mlflow_log
        except ImportError:
            raise ImportError(
                "Please run 'pip install segmind' to install segmind")
        self.segmind = segmind
        self.segmind_log_metrics = log_metrics
        self.segmind_mlflow_log = try_mlflow_log

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
