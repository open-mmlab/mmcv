import datetime

import torch
import torch.distributed as dist

import mmcv
from .base import LoggerHook


class TextLoggerHook(LoggerHook):

    def __init__(self, interval=10, ignore_last=True, reset_flag=False):
        super(TextLoggerHook, self).__init__(interval, ignore_last, reset_flag)
        self.time_sec_tot = 0

    def before_run(self, runner):
        super(TextLoggerHook, self).before_run(runner)
        self.start_iter = runner.iter
        self.dump_log_path = '{}/{}_{}'.format(runner.work_dir,
                                               mmcv.runner.get_time_str(),
                                               'train.json')

    def log(self, runner):
        if runner.mode == 'train':
            lr_str = ', '.join(
                ['{:.5f}'.format(lr) for lr in runner.current_lr()])
            log_str = 'Epoch [{}][{}/{}]\tlr: {}, '.format(
                runner.epoch + 1, runner.inner_iter + 1,
                len(runner.data_loader), lr_str)
        else:
            log_str = 'Epoch({}) [{}][{}]\t'.format(
                runner.mode, runner.epoch + 1, runner.inner_iter)
        # runner.mode is still 'train' at validation
        mode = 'train' if 'time' in runner.log_buffer.output else 'val'
        if 'time' in runner.log_buffer.output:
            self.time_sec_tot += (
                runner.log_buffer.output['time'] * self.interval)
            time_sec_avg = self.time_sec_tot / (
                runner.iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            log_str += 'eta: {}, '.format(eta_str)
            log_str += (
                'time: {log[time]:.3f}, data_time: {log[data_time]:.3f}, '.
                format(log=runner.log_buffer.output))
        log_dict_iter = dict()
        log_dict_iter['mode'] = mode
        log_dict_iter['epoch'] = runner.epoch + 1
        if mode == 'train':
            log_dict_iter['iter'] = runner.inner_iter + 1
            log_dict_iter['lr'] = float(lr_str)
            log_dict_iter['time'] = runner.log_buffer.output['time']
            log_dict_iter['data_time'] = runner.log_buffer.output['data_time']
            # statistic memory
            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated()
                mem_mb = torch.tensor([mem / (1024 * 1024)],
                                      dtype=torch.int,
                                      device=torch.device('cuda'))
                if runner.world_size > 1:
                    dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
                log_str += 'memory: {}, '.format(mem_mb.item())
        log_items = []
        for name, val in runner.log_buffer.output.items():
            if name in ['time', 'data_time']:
                continue
            if isinstance(val, float):
                val = '{:.4f}'.format(val)
            log_items.append('{}: {}'.format(name, val))
            log_dict_iter[name] = val
        log_str += ', '.join(log_items)
        runner.logger.info(log_str)
        if runner.world_size == 1 or (runner.world_size > 1
                                      and dist.get_rank() == 0):
            with open(self.dump_log_path, 'a+') as f:
                mmcv.dump(log_dict_iter, f, file_format='json')
                f.write('\n')
