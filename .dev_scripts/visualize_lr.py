import argparse
import json
import os
import os.path as osp
import time
import warnings
from collections import OrderedDict
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

import mmcv
from mmcv.runner import build_runner
from mmcv.utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the given config'
                                     'of learning rate and '
                                     'momentum')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--work-dir', default='./', help='the dir to save logs and models')
    parser.add_argument(
        '--num-iters', default=300, help='The number of iters per epoch')
    parser.add_argument(
        '--num-epochs', default=300, help='Only used in EpochBasedRinner')
    parser.add_argument(
        '--window-size',
        default='12*14',
        help='Size of the window to display images, in format of "$W*$H".')
    parser.add_argument(
        '--log-interval', default=1, help='The iterval of TextLoggerHook')
    args = parser.parse_args()
    return args


class SimpleModel(nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def train_step(self, *args, **kwargs):
        return dict()

    def val_step(self, *args, **kwargs):
        return dict()


def iter_train(self, data_loader, **kwargs):
    self.mode = 'train'
    self.call_hook('before_train_iter')
    self.call_hook('after_train_iter')
    self._inner_iter += 1
    self._iter += 1


def epoch_train(self, data_loader, **kwargs):
    self.model.train()
    self.mode = 'train'
    self.data_loader = data_loader
    self._max_iters = self._max_epochs * len(self.data_loader)
    self.call_hook('before_train_epoch')
    for i, data_batch in enumerate(self.data_loader):
        self._inner_iter = i
        self.call_hook('before_train_iter')
        self.call_hook('after_train_iter')
        self._iter += 1
    self.call_hook('after_train_epoch')
    self._epoch += 1


def log(self, runner):
    if 'eval_iter_num' in runner.log_buffer.output:
        # this doesn't modify runner.iter and is regardless of by_epoch
        cur_iter = runner.log_buffer.output.pop('eval_iter_num')
    else:
        cur_iter = self.get_iter(runner, inner_iter=True)

    log_dict = OrderedDict(
        mode=self.get_mode(runner),
        epoch=self.get_epoch(runner),
        iter=cur_iter)

    # only record lr of the first param group
    cur_lr = runner.current_lr()
    if isinstance(cur_lr, list):
        log_dict['lr'] = cur_lr[0]
    else:
        assert isinstance(cur_lr, dict)
        log_dict['lr'] = {}
        for k, lr_ in cur_lr.items():
            assert isinstance(lr_, list)
            log_dict['lr'].update({k: lr_[0]})

    if 'time' in runner.log_buffer.output:
        # statistic memory
        if torch.cuda.is_available():
            log_dict['memory'] = self._get_max_memory(runner)

    cur_momentum = runner.current_momentum()
    if isinstance(cur_momentum, list):
        log_dict['momentum'] = cur_momentum[0]
    else:
        assert isinstance(cur_momentum, dict)
        log_dict['momentum'] = {}
        for k, lr_ in cur_momentum.items():
            assert isinstance(lr_, list)
            log_dict['momentum'].update({k: lr_[0]})
    log_dict = dict(log_dict, **runner.log_buffer.output)
    self._log_info(log_dict, runner)
    self._dump_log(log_dict, runner)
    return log_dict


@patch('mmcv.runner.EpochBasedRunner.train', epoch_train)
@patch('mmcv.runner.IterBasedRunner.train', iter_train)
@patch('mmcv.runner.hooks.TextLoggerHook.log', log)
def run(cfg, logger):
    momentum_config = cfg.get('momentum_config')
    lr_config = cfg.get('lr_config')

    model = SimpleModel()
    optimizer = SGD(model.parameters(), 0.1, momentum=0.8)
    cfg.work_dir = cfg.get('work_dir', './')
    workflow = [('train', 1)]

    if cfg.get('runner') is None:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.get('total_epochs', cfg.num_epochs)
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    fake_dataloader = DataLoader(list(range(cfg.num_iters)), batch_size=1)
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=None))
    log_config = dict(
        interval=cfg.log_interval, hooks=[
            dict(type='TextLoggerHook'),
        ])

    runner.register_training_hooks(lr_config, log_config=log_config)
    runner.register_momentum_hook(momentum_config)
    runner.run([fake_dataloader], workflow)


def plot_lr_curve(json_file, cfg):
    data_dict = dict(LearningRate=[], Momentum=[])
    assert os.path.isfile(json_file)
    with open(json_file, 'r') as f:
        for line in f:
            log = json.loads(line.strip())
            data_dict['LearningRate'].append(log['lr'])
            data_dict['Momentum'].append(log['momentum'])

    wind_w, wind_h = [int(size) for size in cfg.window_size.split('*')]
    # if legend is None, use {filename}_{key} as legend
    fig, axes = plt.subplots(2, 1, figsize=(wind_w, wind_h))
    plt.subplots_adjust(hspace=0.5)
    font_size = 20
    for index, (mode, data_list) in enumerate(data_dict.items()):
        ax = axes[index]
        if cfg.runner.type == 'EpochBasedRunner':
            ax.plot(data_list, linewidth=1)
            ax.xaxis.tick_top()
            ax.set_xlabel('Iters', fontsize=font_size)
            ax.xaxis.set_label_position('top')
            sec_ax = ax.secondary_xaxis(
                'bottom',
                functions=(lambda x: x / cfg.num_iters * cfg.log_interval,
                           lambda y: y * cfg.num_iters / cfg.log_interval))
            sec_ax.tick_params(labelsize=font_size)
            sec_ax.set_xlabel('Epochs', fontsize=font_size)
        else:
            # plt.subplot(2, 1, index + 1)
            x_list = np.arange(len(data_list)) * cfg.log_interval
            ax.plot(x_list, data_list)
            ax.set_xlabel('Iters', fontsize=font_size)
        ax.set_ylabel(mode, fontsize=font_size)
        if mode == 'LearningRate':
            if cfg.get('lr_config'):
                title = cfg.lr_config.type
            else:
                title = 'No learning rate schedule'
        else:
            if cfg.get('momentum_config'):
                title = cfg.momentum_config.type
            else:
                title = 'No momentum schedule'
        ax.set_title(title, fontsize=font_size)
        ax.grid()
        # set tick font size
        ax.tick_params(labelsize=font_size)
    save_path = osp.join(cfg.work_dir, 'visualization-result')
    plt.savefig(save_path)
    print(f'The learning rate graph is saved at {save_path}')
    plt.show()


def main():
    args = parse_args()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cfg = mmcv.Config.fromfile(args.config)
    cfg['num_iters'] = args.num_iters
    cfg['num_epochs'] = args.num_epochs
    cfg['log_interval'] = args.log_interval
    cfg['window_size'] = args.window_size

    log_path = osp.join(cfg.get('work_dir', './'), f'{timestamp}.log')
    json_path = log_path + '.json'
    logger = get_logger('mmcv', log_path)

    run(cfg, logger)
    plot_lr_curve(json_path, cfg)


if __name__ == '__main__':
    main()
