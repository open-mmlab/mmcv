import datetime
import os.path as osp
from collections import OrderedDict

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
        self.json_log_path = osp.join(runner.work_dir,
                                      '{}.log.json'.format(runner.timestamp))

    def _get_max_memory(self, runner):
        mem = torch.cuda.max_memory_allocated()
        mem_mb = torch.tensor([mem / (1024 * 1024)],
                              dtype=torch.int,
                              device=torch.device('cuda'))
        if runner.world_size > 1:
            dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        return mem_mb.item()

    def _log_info(self, log_dict, runner):
        if runner.mode == 'train':
            log_str = 'Epoch [{}][{}/{}]\tlr: {:.5f}, '.format(
                log_dict['epoch'], log_dict['iter'], len(runner.data_loader),
                log_dict['lr'])
            if 'time' in log_dict.keys():
                self.time_sec_tot += (log_dict['time'] * self.interval)
                time_sec_avg = self.time_sec_tot / (
                    runner.iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += 'eta: {}, '.format(eta_str)
                log_str += ('time: {:.3f}, data_time: {:.3f}, '.format(
                    log_dict['time'], log_dict['data_time']))
                log_str += 'memory: {}, '.format(log_dict['memory'])
        else:
            log_str = 'Epoch({}) [{}][{}]\t'.format(
                log_dict['mode'], log_dict['epoch'] - 1, log_dict['iter'])
        log_items = []
        for name, val in log_dict.items():
            # TODO: resolve this hack
            # these items have been in log_str
            if name in [
                    'mode', 'Epoch', 'iter', 'lr', 'time', 'data_time',
                    'memory', 'epoch'
            ]:
                continue
            if isinstance(val, float):
                val = '{:.4f}'.format(val)
            log_items.append('{}: {}'.format(name, val))
        log_str += ', '.join(log_items)
        runner.logger.info(log_str)

    def _dump_log(self, log_dict, runner):
        # dump log in json format
        json_log = OrderedDict()
        for k, v in log_dict.items():
            json_log[k] = self._round_float(v)
        # only append log at last line
        if runner.rank == 0:
            with open(self.json_log_path, 'a+') as f:
                mmcv.dump(json_log, f, file_format='json')
                f.write('\n')

    def _round_float(self, items):
        if isinstance(items, list):
            return [self._round_float(item) for item in items]
        elif isinstance(items, float):
            return round(items, 5)
        else:
            return items

    def log(self, runner):
        log_dict = OrderedDict()
        # training mode if the output contains the key "time"
        mode = 'train' if 'time' in runner.log_buffer.output else 'val'
        log_dict['mode'] = mode
        log_dict['epoch'] = runner.epoch + 1
        log_dict['iter'] = runner.inner_iter + 1
        # only record lr of the first param group
        log_dict['lr'] = runner.current_lr()[0]
        if mode == 'train':
            log_dict['time'] = runner.log_buffer.output['time']
            log_dict['data_time'] = runner.log_buffer.output['data_time']
            # statistic memory
            if torch.cuda.is_available():
                log_dict['memory'] = self._get_max_memory(runner)
        for name, val in runner.log_buffer.output.items():
            if name in ['time', 'data_time']:
                continue
            log_dict[name] = val

        self._log_info(log_dict, runner)
        self._dump_log(log_dict, runner)
