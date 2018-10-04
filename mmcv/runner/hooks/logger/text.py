from .base import LoggerHook


class TextLoggerHook(LoggerHook):

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
