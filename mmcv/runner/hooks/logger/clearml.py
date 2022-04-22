# Copyright (c) OpenMMLab. All rights reserved.

from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class ClearMLLoggerHook(LoggerHook):
    """Class to log metrics with clearml.

    It requires `clearml`_ to be installed.


    Args:
        task_init_kwargs (dict): A dict contains the `clearml.Task.init`
        initialization keys. Check
            https://clear.ml/docs/latest/docs/references/sdk/task/#taskinit .
        interval (int): Logging interval (every k iterations). Default 10.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
            Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        by_epoch (bool): Whether EpochBasedRunner is used.
            Default: True.

    .. _clearml:
        https://clear.ml/docs/latest/docs/
    """

    def __init__(self,
                 task_init_kwargs=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True):
        super(ClearMLLoggerHook, self).__init__(interval, ignore_last,
                                                reset_flag, by_epoch)
        self.import_clearml()
        self.task_init_kwargs = task_init_kwargs

    def import_clearml(self):
        try:
            import clearml
        except ImportError:
            raise ImportError(
                'Please run "pip install clearml" to install clearml')
        self.clearml = clearml

    @master_only
    def before_run(self, runner):
        super(ClearMLLoggerHook, self).before_run(runner)
        task_kwargs = self.task_init_kwargs if self.task_init_kwargs else {}
        self.task = self.clearml.Task.init(**task_kwargs)
        self.logger = self.task.get_logger()

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        for tag, val in tags.items():
            self.logger.report_scalar(tag, tag, val, self.get_iter(runner))
