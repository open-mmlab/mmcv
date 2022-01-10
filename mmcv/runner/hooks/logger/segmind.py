# Copyright (c) OpenMMLab. All rights reserved.
from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class SegmindLoggerHook(LoggerHook):

    def __init__(self):
        """Class to log metrics to Segmind.

        It requires `Segmind`_ to be installed.

        .. Segmind:
            https://docs.segmind.com/python-library
        """
        super(SegmindLoggerHook, self).__init__()
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
        self.segmind_log = try_mlflow_log

    @master_only
    def before_run(self, runner):
        super(SegmindLoggerHook, self).before_run(runner)
        if self.segmind is None:
            self.import_segmind()

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if tags:
            # logging metrics to segmind
            self.segmind_log(
                self.log_metrics, tags, step=runner.epoch, epoch=runner.epoch)

    @master_only
    def after_run(self, runner):
        pass
