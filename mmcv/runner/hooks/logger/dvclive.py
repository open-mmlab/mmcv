# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path

from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class DvcliveLoggerHook(LoggerHook):
    """Class to log metrics with dvclive.

    It requires `dvclive`_ to be installed.

    Args:
        model_file (str): Default None. If not None, after each epoch the
            model will be saved to {model_file}.
        interval (int): Logging interval (every k iterations). Default 10.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        by_epoch (bool): Whether EpochBasedRunner is used. Default: True.
        kwargs: Arguments for instantiating `Live`_.

    .. _dvclive:
        https://dvc.org/doc/dvclive

    .. _Live:
        https://dvc.org/doc/dvclive/api-reference/live#parameters
    """

    def __init__(self,
                 model_file=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True,
                 **kwargs):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.model_file = model_file
        self.import_dvclive(**kwargs)

    def import_dvclive(self, **kwargs):
        try:
            from dvclive import Live
        except ImportError:
            raise ImportError(
                'Please run "pip install dvclive" to install dvclive')
        self.dvclive = Live(**kwargs)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if tags:
            self.dvclive.set_step(self.get_iter(runner))
            for k, v in tags.items():
                self.dvclive.log(k, v)

    @master_only
    def after_train_epoch(self, runner):
        super().after_train_epoch(runner)
        if self.model_file is not None:
            runner.save_checkpoint(
                Path(self.model_file).parent,
                filename_tmpl=Path(self.model_file).name,
                create_symlink=False,
            )
