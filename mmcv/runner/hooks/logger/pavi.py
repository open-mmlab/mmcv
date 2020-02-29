# Copyright (c) Open-MMLab. All rights reserved.
from ...dist_utils import master_only
from .base import LoggerHook


class PaviLoggerHook(LoggerHook):

    def __init__(self,
                 init_kwargs=dict(),
                 interval=10,
                 ignore_last=True,
                 reset_flag=True):
        super(PaviLoggerHook, self).__init__(interval, ignore_last, reset_flag)
        self.init_kwargs = init_kwargs

    @master_only
    def before_run(self, runner):
        try:
            from pavi import SummaryWriter
        except ImportError:
            raise ImportError('Please run "pip install pavi" to install pavi.')
        self.writer = SummaryWriter(**self.init_kwargs)

    @master_only
    def log(self, runner):
        tags = {}
        for tag, val in runner.log_buffer.output.items():
            if tag in ['time', 'data_time']:
                continue
            tags[tag] = val
        if tags:
            self.writer.add_scalars(runner.mode, tags, runner.iter)
