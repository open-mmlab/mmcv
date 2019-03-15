import os.path as osp

from .base import LoggerHook
from ...utils import master_only


class TensorboardLoggerHook(LoggerHook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True):
        super(TensorboardLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag)
        self.log_dir = log_dir

    @master_only
    def before_run(self, runner):
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError('Please install tensorflow and tensorboardX '
                              'to use TensorboardLoggerHook.')
        else:
            if self.log_dir is None:
                self.log_dir = osp.join(runner.work_dir, 'tf_logs')
            self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        for var in runner.log_buffer.output:
            if var in ['time', 'data_time']:
                continue
            tag = '{}/{}'.format(var, runner.mode)
            record = runner.log_buffer.output[var]
            if isinstance(record, str):
                self.writer.add_text(tag, record, runner.iter)
            else:
                self.writer.add_scalar(tag, runner.log_buffer.output[var],
                                       runner.iter)

    @master_only
    def after_run(self, runner):
        self.writer.close()
