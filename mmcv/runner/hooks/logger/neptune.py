# Copyright (c) Open-MMLab. All rights reserved.
from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class NeptuneLoggerHook(LoggerHook):

    def __init__(self,
                 init_kwargs=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 with_step=True,
                 by_epoch=True):
        super(NeptuneLoggerHook, self).__init__(interval, ignore_last,
                                                reset_flag, by_epoch)
        self.import_neptune()
        self.init_kwargs = init_kwargs
        self.with_step = with_step

    def import_neptune(self):
        try:
            import neptune.new as neptune
        except ImportError:
            raise ImportError(
                'Please run "pip install neptune-client" to install neptune')
        self.neptune = neptune
        self.run = None

    @master_only
    def before_run(self, runner):
        if self.neptune is None:
            self.import_neptune()
        if self.init_kwargs:
            self.run = self.neptune.init(**self.init_kwargs)
        else:
            self.run = self.neptune.init()

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if tags:
            for tag_name, tag_value in tags.items():
                if self.with_step:
                    print(tag_name, tag_value)
                    self.run[tag_name].log(
                        tag_value, step=self.get_iter(runner))
                else:
                    tags['global_step'] = self.get_iter(runner)
                    self.run[tag_name].log(tags)

    @master_only
    def after_run(self, runner):
        self.run.stop()
