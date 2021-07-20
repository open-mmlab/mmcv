# Copyright (c) Open-MMLab. All rights reserved.
from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class DvcliveLoggerHook(LoggerHook):
    """Class to log metrics with dvclive.

    It requires `dvclive`_ to be installed.

    Args:
        interval (int): Logging interval (every k iterations).
            Default 10.
            If `by_epoch` is True, the value will be set to 0 in
            order to properly work with `dvclive`_.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
            Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: True.
        by_epoch (bool): Whether EpochBasedRunner is used.
            Default: True.

    .. _dvclive:
        https://dvc.org/doc/dvclive
    """

    def __init__(self,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 by_epoch=True):
        if by_epoch:
            interval = 0
        super(DvcliveLoggerHook, self).__init__(interval, ignore_last,
                                                reset_flag, by_epoch)
        self.import_dvclive()

    def import_dvclive(self):
        try:
            import dvclive
        except ImportError:
            raise ImportError(
                'Please run "pip install dvclive" to install dvclive')
        self.dvclive = dvclive

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if self.by_epoch:
            step = self.get_epoch(runner)
        else:
            step = self.get_iter(runner)
        if tags:
            for k, v in tags.items():
                self.dvclive.log(k, v, step=step)

    @master_only
    def after_train_epoch(self, runner):
        super().after_train_epoch(runner)
        runner.save_checkpoint(
            runner.work_dir, filename_tmpl='model.pth', create_symlink=False)
        self.dvclive.next_step()
