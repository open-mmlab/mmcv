# Copyright (c) OpenMMLab. All rights reserved.
from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class SMExperimentsLoggerHook(LoggerHook):
    """Class to log metrics to Sagemaker Experiments. Works within sagemaker
    training job only due to library implementation. It requires `sagemaker-
    experiments`_ and `boto3`_ to be installed.

    Args:
        interval (int): Logging interval (every k iterations). Default: 10.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        by_epoch (bool): Whether EpochBasedRunner is used. Default: True.

    .. _sagemaker-experiments:
        https://sagemaker-experiments.readthedocs.io/
    .. _boto3:
        https://boto3.readthedocs.io/
    """

    def __init__(self,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True,
                 init_kwargs=None):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.import_smexperiments()
        self.tracker = None
        self.init_kwargs = init_kwargs

    def import_smexperiments(self):
        try:
            import boto3
            from smexperiments import tracker
        except ImportError:
            raise ImportError("""Please run
                    "pip install sagemaker-experiments boto3"
                    to use smexperiments""")
        self.smexp_tracker = tracker
        self.boto3 = boto3

    def _create_tracker(self, **kwargs):
        self.tracker = self.smexp_tracker.Tracker.load(**kwargs)

    @master_only
    def before_run(self, runner):
        super().before_run(runner)
        tracker_kwargs = self.init_kwargs if self.init_kwargs else {}
        self._create_tracker(**tracker_kwargs)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if tags:
            for tag_name, tag_value in tags.items():
                self.tracker.log_metric(
                    tag_name,
                    tag_value,
                    iteration_number=self.get_iter(runner))

    @master_only
    def after_run(self, runner):
        if self.tracker:
            self.tracker.close()
