import datetime

from .base import LoggerHook


class TextLoggerHook(LoggerHook):

    def __init__(self, interval=10, ignore_last=True, reset_flag=False):
        super(TextLoggerHook, self).__init__(interval, ignore_last, reset_flag)
        self.time_sec_tot = 0

    def log(self, runner):
        if runner.mode == 'train':
            lr_str = ', '.join(
                ['{:.5f}'.format(lr) for lr in runner.current_lr()])
            log_str = 'Epoch [{}][{}/{}]\tlr: {}, '.format(
                runner.epoch + 1, runner.inner_iter + 1,
                len(runner.data_loader), lr_str)
        else:
            log_str = 'Epoch({}) [{}][{}]\t'.format(runner.mode, runner.epoch,
                                                    runner.inner_iter + 1)
        if 'time' in runner.log_buffer.output:
            self.time_sec_tot += (runner.log_buffer.output['time'] *
                                  self.interval)
            time_sec_avg = self.time_sec_tot / (runner.iter + 1)
            eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            log_str += 'eta: {}, '.format(eta_str)
            log_str += (
                'time: {log[time]:.3f}, data_time: {log[data_time]:.3f}, '.
                format(log=runner.log_buffer.output))
        log_items = []
        for name, val in runner.log_buffer.output.items():
            if name in ['time', 'data_time']:
                continue
            log_items.append('{}: {:.4f}'.format(name, val))
        log_str += ', '.join(log_items)
        runner.logger.info(log_str)
