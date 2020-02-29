# Copyright (c) Open-MMLab. All rights reserved.
from ...dist_utils import master_only
from .base import LoggerHook


class PaviLoggerHook(LoggerHook):

    def __init__(self,
                 init_kwargs=None,
                 add_graph=False,
                 add_last_ckpt=False,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True):
        super(PaviLoggerHook, self).__init__(interval, ignore_last, reset_flag)
        self.init_kwargs = init_kwargs
        self.add_graph = add_graph
        self.add_last_ckpt = add_last_ckpt

    @master_only
    def before_run(self, runner):
        try:
            from pavi import SummaryWriter
        except ImportError:
            raise ImportError('Please run "pip install pavi" to install pavi.')

        self.run_name = runner.work_dir.split('/')[-1]

        if not self.init_kwargs:
            self.init_kwargs = dict()
        if 'task' not in self.init_kwargs.keys():
            self.init_kwargs['task'] = self.run_name
        if 'taskid' not in self.init_kwargs.keys():
            self.init_kwargs['taskid'] = self.init_kwargs['task']
        if 'model' not in self.init_kwargs.keys():
            self.init_kwargs['model'] = runner._model_name

        self.writer = SummaryWriter(**self.init_kwargs)

        if self.add_graph:
            self.writer.add_graph(runner.model)

    @master_only
    def after_run(self, runner):
        if self.add_last_ckpt:
            self.writer.add_torch_snapshot(
                tag=self.run_name,
                snapshot=runner.checkpoint,
                iteration=runner.iter)

    @master_only
    def log(self, runner):
        tags = {}
        for tag, val in runner.log_buffer.output.items():
            if tag in ['time', 'data_time']:
                continue
            tags[tag] = val
        if tags:
            self.writer.add_scalars(runner.mode, tags, runner.iter)
